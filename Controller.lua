--!native
--!optimize 2
-- Native code generation + aggressive optimization: compiles the numeric hot
-- path (accumulation, reconstruction, dispatch) to machine code instead of interpreted bytecode.

local AssetService      = game:GetService("AssetService")
local Players           = game:GetService("Players")
local UserInputService  = game:GetService("UserInputService")
local Lighting          = game:GetService("Lighting")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService        = game:GetService("RunService")

local player = Players.LocalPlayer
local camera = workspace.CurrentCamera

local TILE_W  = 960
local TILE_H  = 540
local FULL_W  = TILE_W * 2
local FULL_H  = TILE_H * 2
local CHUNK_W = 60
local CHUNK_H = 60

local NUM_WORKERS   = 8
local NUM_TILES     = 4
local DISPLAY_EVERY = 1

-- ─── Upscaling constants ───────────────────────────────────────────────────
-- Internal render resolution (half of output)
local INT_TILE_W = 480
local INT_TILE_H = 270
local INT_FULL_W = INT_TILE_W * 2
local INT_FULL_H = INT_TILE_H * 2
local UPSCALE_ENABLED = true  -- toggled from settings UI

-- ─── Halton low-discrepancy sequence ───────────────────────────────────────
-- Used for subpixel jitter in ASR: shifts the internal sample grid by a
-- fraction of an internal pixel each pass so accumulated samples cover the
-- full output resolution space rather than the same coarse grid every pass.

local function halton(index, base)
	local result = 0
	local f = 1
	local i = index
	while i > 0 do
		f = f / base
		result = result + f * (i % base)
		i = math.floor(i / base)
	end
	return result
end

-- HDR encoding ceiling. MUST match the worker's HDR_MAX and the worker's
-- per-sample clamp (maxVal). Worker encodes value/HDR_MAX*255, controller
-- decodes byte/255*HDR_MAX. Mismatch scales all geometry color.
local HDR_MAX = 8.0

-- Precomputed dither LUT (D): the 8-bit write dither was recomputed per pixel
-- as ((p*2654435761)%256)/255-0.5 — a multiply, mod, divide, subtract on the
-- single most-run line in the renderer. It only depends on the tile-pixel
-- index p, so bake it once into an f32 buffer and read it back. Indexed by
-- p = y*TILE_W + x (same layout every tile), so one table serves all tiles.
local DITHER_BUF = buffer.create(TILE_W * TILE_H * 4)
for p = 0, TILE_W * TILE_H - 1 do
	buffer.writef32(DITHER_BUF, p * 4, ((p * 2654435761) % 256) / 255 - 0.5)
end

-- ─── Settings & presets ────────────────────────────────────────────────────

local settings = {
	showPlayer     = false,
	exposure       = 0.5,
	bloomThreshold = 1.0,
	bloomIntensity = 0.40,
	bounceStrength = 0.6,
	sunSampleProb  = 0.25,
	aoRays         = 3,
	maxBounces     = 3,
	rtResIndex     = 1,   -- realtime resolution preset (1=lowest … 6=highest); 1 = 128x72
	rtSamples      = 1,   -- realtime samples per pixel per frame (noise↓ while moving, fps↓)
	rtBounces      = 1,   -- realtime GI bounces (1–3); more = GI but noisier/slower
	rtPixelated    = true,  -- realtime display scaling: false=bilinear, true=pixelated
	rtWindowMode   = 2,     -- realtime window: 1=fullscreen, 2=windowed, 3=1:1 native
	rtRayTrace     = true,  -- realtime: ray-tracing mode (no GI bounce → clean, no noise, faster) [default]
}

local presets = {
	{
		name           = "Balanced",
		maxBounces     = 2,
		exposure       = 0.50,
		bloomThreshold = 1.0,
		bloomIntensity = 0.30,
		bounceStrength = 0.60,
		sunSampleProb  = 0.25,
		aoRays         = 3,
	},
	{
		name           = "Detailed",
		maxBounces     = 3,
		exposure       = 0.50,
		bloomThreshold = 1.0,
		bloomIntensity = 0.25,
		bounceStrength = 0.65,
		sunSampleProb  = 0.15,
		aoRays         = 6,
	},
	{
		name           = "Global Illumination",
		maxBounces     = 5,
		exposure       = 0.50,
		bloomThreshold = 1.2,
		bloomIntensity = 0.20,
		bounceStrength = 0.85,
		sunSampleProb  = 0.08,
		aoRays         = 6,
	},
	{
		name           = "Realistic",
		maxBounces     = 5,
		exposure       = 0.45,
		bloomThreshold = 1.6,
		bloomIntensity = 0.10,
		bounceStrength = 0.72,
		sunSampleProb  = 0.06,
		aoRays         = 8,
	},
	{
		name           = "Cinematic",
		maxBounces     = 5,
		exposure       = 0.48,
		bloomThreshold = 1.4,
		bloomIntensity = 0.18,
		bounceStrength = 0.78,
		sunSampleProb  = 0.06,
		aoRays         = 8,
	},
	{
		name           = "Reference",
		maxBounces     = 6,
		exposure       = 0.45,
		bloomThreshold = 1.8,
		bloomIntensity = 0.08,
		bounceStrength = 0.80,
		sunSampleProb  = 0.04,
		aoRays         = 12,
	},
}

-- ─── Bloom constants ───────────────────────────────────────────────────────

local BLOOM_W2 = TILE_W // 2
local BLOOM_H2 = TILE_H // 2
local BLOOM_W4 = TILE_W // 4
local BLOOM_H4 = TILE_H // 4

local GAUSS5 = {}
do
	local sigma = 1.5
	local sum   = 0
	for k = 1, 5 do
		local x = k - 3
		GAUSS5[k] = math.exp(-(x * x) / (2 * sigma * sigma))
		sum += GAUSS5[k]
	end
	for k = 1, 5 do GAUSS5[k] = GAUSS5[k] / sum end
end

local BLOOM_UX0, BLOOM_UX1, BLOOM_UTX = {}, {}, {}
local BLOOM_UY0, BLOOM_UY1, BLOOM_UTY = {}, {}, {}
for x = 0, TILE_W - 1 do
	local fx = math.max(0, x * 0.5 - 0.25)
	local x0 = math.min(math.floor(fx), BLOOM_W2 - 1)
	BLOOM_UX0[x] = x0
	BLOOM_UX1[x] = math.min(x0 + 1, BLOOM_W2 - 1)
	BLOOM_UTX[x] = fx - math.floor(fx)
end
for y = 0, TILE_H - 1 do
	local fy = math.max(0, y * 0.5 - 0.25)
	local y0 = math.min(math.floor(fy), BLOOM_H2 - 1)
	BLOOM_UY0[y] = y0
	BLOOM_UY1[y] = math.min(y0 + 1, BLOOM_H2 - 1)
	BLOOM_UTY[y] = fy - math.floor(fy)
end

-- ─── Tonemap ───────────────────────────────────────────────────────────────

local function acesTonemap(x)
	local a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
	return math.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
end
local GAMMA_INV = 1 / 2.2
local function tonemapChannel(v)
	return acesTonemap(math.max(0, v) * settings.exposure) ^ GAMMA_INV
end
local function dwrite(v, noise)
	return math.clamp(math.floor(tonemapChannel(v) * 255 + noise), 0, 255)
end

-- Sky pixels bypass ACES — values are [0,1], apply exposure + gamma only
local function dwriteSky(v, noise)
	local linear = math.max(0, v) * settings.exposure
	local gamma  = math.min(1, linear) ^ GAMMA_INV
	return math.clamp(math.floor(gamma * 255 + noise), 0, 255)
end

-- ─── ScreenGui ─────────────────────────────────────────────────────────────

local screenGui = script.Parent
screenGui.IgnoreGuiInset = true
screenGui.ResetOnSpawn   = false
screenGui.Enabled        = true

-- ─── HUD ScreenGui ────────────────────────────────────────────────────────

local hudGui = Instance.new("ScreenGui")
hudGui.Name            = "RayTracerHUD"
hudGui.IgnoreGuiInset  = true
hudGui.ResetOnSpawn    = false
hudGui.ZIndexBehavior  = Enum.ZIndexBehavior.Sibling
hudGui.Enabled         = true
hudGui.Parent          = player.PlayerGui

local passLabel = Instance.new("TextLabel")
passLabel.Size                   = UDim2.new(0, 220, 0, 30)
passLabel.Position               = UDim2.new(0.5, -110, 0, 10)
passLabel.BackgroundColor3       = Color3.fromRGB(0, 0, 0)
passLabel.BackgroundTransparency = 0.5
passLabel.TextColor3             = Color3.fromRGB(255, 255, 255)
passLabel.Font                   = Enum.Font.GothamBold
passLabel.TextSize               = 14
passLabel.Text                   = "Pass 0"
passLabel.Visible                = true
passLabel.ZIndex                 = 10
passLabel.Parent                 = screenGui

-- ─── Settings panel ────────────────────────────────────────────────────────

local PANEL_W        = 320
local TRACK_W        = PANEL_W - 28
local TRACK_H        = 3
local THUMB_SIZE     = 11
local ROW_H          = 38   -- height per slider row
local SECTION_H      = 22   -- height per section header
local TOGGLE_ROW_H   = 26

-- Layout: presets (58px) + 3 sections with sliders + toggle rows + padding
-- RENDER: exposure, bounces, bounce strength, sun sampling, ao rays (5 sliders)
-- POST:   bloom intensity, bloom threshold (2 sliders)
-- ASR:    rt.res.step, rt.samples, rt.window (3 sliders) + 3 toggles
local PANEL_H = 58 + (SECTION_H + 5*ROW_H) + 8 + (SECTION_H + 2*ROW_H) + 8 + (SECTION_H + 4*ROW_H + 4*TOGGLE_ROW_H) + 16

-- Visible height is clamped so a tall panel scrolls instead of running off screen.
local vpY        = (workspace.CurrentCamera and workspace.CurrentCamera.ViewportSize.Y) or 720
local PANEL_VIEW = math.min(PANEL_H, math.max(240, vpY - 110))

local settingsPanel = Instance.new("ScrollingFrame")
settingsPanel.Size                   = UDim2.new(0, PANEL_W, 0, PANEL_VIEW)
settingsPanel.Position               = UDim2.new(1, -(PANEL_W + 8), 1, -(PANEL_VIEW + 48))
settingsPanel.BackgroundColor3       = Color3.fromRGB(8, 8, 12)
settingsPanel.BackgroundTransparency = 0.05
settingsPanel.BorderSizePixel        = 0
settingsPanel.Visible                = false
settingsPanel.CanvasSize             = UDim2.new(0, 0, 0, PANEL_H)   -- full content height
settingsPanel.ScrollingDirection     = Enum.ScrollingDirection.Y
settingsPanel.ScrollBarThickness     = 4
settingsPanel.ScrollBarImageColor3   = Color3.fromRGB(74, 158, 255)
settingsPanel.CanvasPosition         = Vector2.new(0, 0)
settingsPanel.Active                 = true
settingsPanel.Parent                 = hudGui

-- Remembered scroll position across close/open.
local savedScroll = Vector2.new(0, 0)

-- Left accent bar
local accentBar = Instance.new("Frame")
accentBar.Size             = UDim2.new(0, 2, 1, 0)
accentBar.BackgroundColor3 = Color3.fromRGB(74, 158, 255)
accentBar.BorderSizePixel  = 0
accentBar.Parent           = settingsPanel

-- Preset row
local presetContainer = Instance.new("Frame")
presetContainer.Size                   = UDim2.new(1, -12, 0, 50)
presetContainer.Position               = UDim2.new(0, 10, 0, 8)
presetContainer.BackgroundTransparency = 1
presetContainer.Parent                 = settingsPanel
do
	local gl = Instance.new("UIGridLayout")
	gl.CellSize              = UDim2.new(0, 97, 0, 22)
	gl.CellPadding           = UDim2.new(0, 3, 0, 3)
	gl.FillDirectionMaxCells = 3
	gl.Parent                = presetContainer
end

local presetButtons = {}
local function applyPreset(preset)
	for k, v in pairs(preset) do
		if k ~= "name" then settings[k] = v end
	end
end

for _, preset in ipairs(presets) do
	local btn = Instance.new("TextButton")
	btn.Size             = UDim2.new(0, 90, 0, 22)
	btn.BackgroundColor3 = Color3.fromRGB(20, 20, 28)
	btn.BorderSizePixel  = 0
	btn.TextColor3       = Color3.fromRGB(130, 130, 148)
	btn.Font             = Enum.Font.GothamMedium
	btn.TextSize         = 9
	btn.Text             = string.upper(preset.name)
	btn.Parent           = presetContainer
	local cap = preset
	btn.MouseButton1Click:Connect(function()
		applyPreset(cap)
		for _, b in ipairs(presetButtons) do
			b.BackgroundColor3 = Color3.fromRGB(20, 20, 28)
			b.TextColor3       = Color3.fromRGB(130, 130, 148)
		end
		btn.BackgroundColor3 = Color3.fromRGB(18, 52, 96)
		btn.TextColor3       = Color3.fromRGB(74, 158, 255)
	end)
	table.insert(presetButtons, btn)
end

-- Section header helper
local function makeSection(label, yPos)
	local line = Instance.new("Frame")
	line.Size             = UDim2.new(1, -12, 0, 1)
	line.Position         = UDim2.new(0, 10, 0, yPos + 9)
	line.BackgroundColor3 = Color3.fromRGB(25, 25, 35)
	line.BorderSizePixel  = 0
	line.Parent           = settingsPanel
	local lbl = Instance.new("TextLabel")
	lbl.Size                   = UDim2.new(0, 80, 0, SECTION_H)
	lbl.Position               = UDim2.new(0, 10, 0, yPos)
	lbl.BackgroundTransparency = 1
	lbl.TextColor3             = Color3.fromRGB(74, 158, 255)
	lbl.Font                   = Enum.Font.GothamBold
	lbl.TextSize               = 10
	lbl.TextXAlignment         = Enum.TextXAlignment.Left
	lbl.Text                   = label
	lbl.Parent                 = settingsPanel
end

-- Slider definitions split into sections
local sliderSections = {
	{
		name = "— RENDER",
		sliders = {
			{ key = "exposure",       label = "exposure",        min = 0.1, max = 2.0, fmt = "%.2f" },
			{ key = "maxBounces",     label = "bounces",         min = 1,   max = 6,   fmt = "%d", isInt = true },
			{ key = "bounceStrength", label = "bounce.strength", min = 0.0, max = 1.5, fmt = "%.2f" },
			{ key = "sunSampleProb",  label = "sun.sampling",    min = 0.0, max = 1.0, fmt = "%.2f" },
			{ key = "aoRays",         label = "ao.rays",         min = 0,   max = 12,  fmt = "%d", isInt = true },
		},
	},
	{
		name = "— POST",
		sliders = {
			{ key = "bloomIntensity", label = "bloom.intensity", min = 0.0, max = 1.0, fmt = "%.2f" },
			{ key = "bloomThreshold", label = "bloom.threshold", min = 0.1, max = 3.0, fmt = "%.2f" },
		},
	},
	{
		name = "— ASR",
		sliders = {
			{ key = "rtResIndex", label = "rt.res.step",  min = 1, max = 6,   fmt = "%d", isInt = true },
			{ key = "rtSamples",  label = "rt.samples",   min = 1, max = 8,   fmt = "%d", isInt = true },
			{ key = "rtBounces",  label = "rt.bounces",   min = 1, max = 3,   fmt = "%d", isInt = true },
			{ key = "rtWindowMode", label = "rt.window",  min = 1, max = 3,   fmt = "%d", isInt = true },
		},
	},
}

-- Flat list of all sliders for refresh
local refreshSliders = {}
local allSliderDefs  = {}
for _, section in ipairs(sliderSections) do
	for _, def in ipairs(section.sliders) do
		table.insert(allSliderDefs, def)
	end
end
-- keep sliderDefs alias for anything that references it later
local sliderDefs = allSliderDefs

local function makeSlider(def, yPos)
	local lbl = Instance.new("TextLabel")
	lbl.Size                   = UDim2.new(0, TRACK_W - 50, 0, 14)
	lbl.Position               = UDim2.new(0, 14, 0, yPos)
	lbl.BackgroundTransparency = 1
	lbl.TextColor3             = Color3.fromRGB(180, 180, 200)
	lbl.Font                   = Enum.Font.Code
	lbl.TextSize               = 11
	lbl.TextXAlignment         = Enum.TextXAlignment.Left
	lbl.Text                   = def.label
	lbl.Parent                 = settingsPanel

	local valLbl = Instance.new("TextLabel")
	valLbl.Size                   = UDim2.new(0, 48, 0, 14)
	valLbl.Position               = UDim2.new(0, PANEL_W - 60, 0, yPos)
	valLbl.BackgroundTransparency = 1
	valLbl.TextColor3             = Color3.fromRGB(100, 180, 255)
	valLbl.Font                   = Enum.Font.Code
	valLbl.TextSize               = 11
	valLbl.TextXAlignment         = Enum.TextXAlignment.Right
	valLbl.Parent                 = settingsPanel

	local track = Instance.new("Frame")
	track.Size             = UDim2.new(0, TRACK_W, 0, TRACK_H)
	track.Position         = UDim2.new(0, 14, 0, yPos + 17)
	track.BackgroundColor3 = Color3.fromRGB(25, 25, 35)
	track.BorderSizePixel  = 0
	track.Parent           = settingsPanel
	do
		local c2 = Instance.new("UICorner")
		c2.CornerRadius = UDim.new(1, 0)
		c2.Parent = track
	end

	local fill = Instance.new("Frame")
	fill.Size             = UDim2.new(0, 0, 1, 0)
	fill.BackgroundColor3 = Color3.fromRGB(74, 158, 255)
	fill.BorderSizePixel  = 0
	fill.Parent           = track
	do
		local c2 = Instance.new("UICorner")
		c2.CornerRadius = UDim.new(1, 0)
		c2.Parent = fill
	end

	local thumb = Instance.new("Frame")
	thumb.Size             = UDim2.new(0, THUMB_SIZE, 0, THUMB_SIZE)
	thumb.AnchorPoint      = Vector2.new(0.5, 0.5)
	thumb.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
	thumb.BorderSizePixel  = 0
	thumb.ZIndex           = 5
	thumb.Parent           = track
	do
		local c2 = Instance.new("UICorner")
		c2.CornerRadius = UDim.new(1, 0)
		c2.Parent = thumb
	end

	local function refresh()
		local val = settings[def.key]
		local t   = math.clamp((val - def.min) / (def.max - def.min), 0, 1)
		fill.Size      = UDim2.new(t, 0, 1, 0)
		thumb.Position = UDim2.new(t, 0, 0.5, 0)
		valLbl.Text    = string.format(def.fmt, val)
	end
	refreshSliders[def.key] = refresh

	local dragging = false
	local function setFromX(absX)
		local t   = math.clamp((absX - track.AbsolutePosition.X) / track.AbsoluteSize.X, 0, 1)
		local raw = def.min + t * (def.max - def.min)
		settings[def.key] = def.isInt and math.round(raw) or raw
		refresh()
	end

	thumb.InputBegan:Connect(function(i)
		if i.UserInputType == Enum.UserInputType.MouseButton1 then dragging = true end
	end)
	track.InputBegan:Connect(function(i)
		if i.UserInputType == Enum.UserInputType.MouseButton1 then
			dragging = true
			setFromX(i.Position.X)
		end
	end)
	UserInputService.InputChanged:Connect(function(i)
		if dragging and i.UserInputType == Enum.UserInputType.MouseMovement then
			setFromX(i.Position.X)
		end
	end)
	UserInputService.InputEnded:Connect(function(i)
		if i.UserInputType == Enum.UserInputType.MouseButton1 then dragging = false end
	end)

	refresh()
end

-- Build sections with proper Y positioning
local renderData = {}  -- forward declare so toggle can update it
local curY = 58  -- start below presets

local function makeToggleRow(labelText, yPos, initState, onToggle)
	local lbl = Instance.new("TextLabel")
	lbl.Size                   = UDim2.new(0, TRACK_W - 40, 0, TOGGLE_ROW_H)
	lbl.Position               = UDim2.new(0, 14, 0, yPos)
	lbl.BackgroundTransparency = 1
	lbl.TextColor3             = Color3.fromRGB(180, 180, 200)
	lbl.Font                   = Enum.Font.Code
	lbl.TextSize               = 11
	lbl.TextXAlignment         = Enum.TextXAlignment.Left
	lbl.Text                   = labelText
	lbl.Parent                 = settingsPanel

	local track2 = Instance.new("TextButton")
	track2.Size             = UDim2.new(0, 32, 0, 16)
	track2.Position         = UDim2.new(0, PANEL_W - 46, 0, yPos + 5)
	track2.BackgroundColor3 = initState and Color3.fromRGB(18, 52, 96) or Color3.fromRGB(20, 20, 28)
	track2.BorderSizePixel  = 0
	track2.Text             = ""
	track2.Parent           = settingsPanel
	do local c2=Instance.new("UICorner"); c2.CornerRadius=UDim.new(1,0); c2.Parent=track2 end

	local dot = Instance.new("Frame")
	dot.Size             = UDim2.new(0, 11, 0, 11)
	dot.Position         = initState and UDim2.new(1, -13, 0.5, -5.5) or UDim2.new(0, 2, 0.5, -5.5)
	dot.BackgroundColor3 = initState and Color3.fromRGB(74, 158, 255) or Color3.fromRGB(50, 50, 65)
	dot.BorderSizePixel  = 0
	dot.Parent           = track2
	do local c2=Instance.new("UICorner"); c2.CornerRadius=UDim.new(1,0); c2.Parent=dot end

	local state = {value = initState}
	track2.MouseButton1Click:Connect(function()
		state.value = not state.value
		track2.BackgroundColor3 = state.value and Color3.fromRGB(18, 52, 96) or Color3.fromRGB(20, 20, 28)
		dot.BackgroundColor3    = state.value and Color3.fromRGB(74, 158, 255) or Color3.fromRGB(50, 50, 65)
		dot.Position            = state.value and UDim2.new(1, -13, 0.5, -5.5) or UDim2.new(0, 2, 0.5, -5.5)
		onToggle(state.value)
	end)
	return state, track2, dot
end

for _, section in ipairs(sliderSections) do
	makeSection(section.name, curY)
	curY += SECTION_H + 2
	for _, def in ipairs(section.sliders) do
		makeSlider(def, curY)
		curY += ROW_H
	end
	curY += 8
end

-- ASR toggles (appended to ASR section)
curY -= 8  -- remove last section gap, we're still in ASR

local asrState, _, _ = makeToggleRow(
	"asr." .. INT_TILE_W .. "p→" .. TILE_W .. "p  (" .. (TILE_W/INT_TILE_W) .. "x)",
	curY, UPSCALE_ENABLED,
	function(v) UPSCALE_ENABLED = v end
)
curY += TOGGLE_ROW_H

-- Realtime display scaling: off = bilinear (smooth/blurry), on = pixelated (crisp/blocky).
-- Just flips the setting; the realtime loop applies it to the images live.
local rtPixelState = makeToggleRow(
	"rt.pixelated",
	curY, settings.rtPixelated,
	function(v) settings.rtPixelated = v end
)
curY += TOGGLE_ROW_H

-- Realtime ray-tracing mode: drop the stochastic GI bounce for a clean, noiseless,
-- deterministic image (sharp direct light, reflections, boosted ambient) — faster too.
local rtRayState = makeToggleRow(
	"rt.raytrace",
	curY, settings.rtRayTrace,
	function(v) settings.rtRayTrace = v end
)
curY += TOGGLE_ROW_H

local showPlayerBtn, playerToggleTrack, playerToggleDot
local playerState, playerTrack, playerDot = makeToggleRow(
	"show.player",
	curY, settings.showPlayer,
	function(v)
		settings.showPlayer = v
		local newFilter = {}
		if not settings.showPlayer and player.Character then
			table.insert(newFilter, player.Character)
		end
		renderData.characterFilter = newFilter
	end
)
curY += TOGGLE_ROW_H + 8

-- Aliases for existing code
local toggleBtn = playerTrack
local upscaleBtn = playerTrack  -- placeholder, not used by text anymore
local function updateToggle() end  -- no-op, toggle is self-contained now

-- ─── HUD buttons ───────────────────────────────────────────────────────────

local BTN_SIZE = 30
local function makeBtn(text, xFromRight)
	local btn = Instance.new("TextButton")
	btn.Size                   = UDim2.new(0, BTN_SIZE, 0, BTN_SIZE)
	btn.Position               = UDim2.new(1, xFromRight, 1, -8 - BTN_SIZE)
	btn.BackgroundColor3       = Color3.fromRGB(10, 10, 15)
	btn.BackgroundTransparency = 0.1
	btn.BorderSizePixel        = 0
	btn.TextColor3             = Color3.fromRGB(150, 150, 165)
	btn.Font                   = Enum.Font.GothamBold
	btn.TextSize               = 14
	btn.Text                   = text
	btn.Parent                 = hudGui
	return btn
end

local settingsBtn = makeBtn("⚙", -(BTN_SIZE + 8))
settingsBtn.MouseButton1Click:Connect(function()
	if settingsPanel.Visible then
		savedScroll = settingsPanel.CanvasPosition   -- remember where we were
		settingsPanel.Visible = false
	else
		settingsPanel.Visible = true
		for _, fn in pairs(refreshSliders) do fn() end
		settingsPanel.CanvasPosition = savedScroll    -- restore scroll
	end
end)

local uiHidden = false

-- ─── Tiles & workers ───────────────────────────────────────────────────────

local actorTemplate = ReplicatedStorage:WaitForChild("RenderActor")

local tileOffsets = {
	{x = 0,      y = 0,      ix = 0,          iy = 0         },
	{x = TILE_W, y = 0,      ix = INT_TILE_W, iy = 0         },
	{x = 0,      y = TILE_H, ix = 0,          iy = INT_TILE_H},
	{x = TILE_W, y = TILE_H, ix = INT_TILE_W, iy = INT_TILE_H},
}

local tiles   = {}
local workers = {}

for i = 1, NUM_TILES do
	local offset = tileOffsets[i]

	local frame = Instance.new("Frame")
	frame.Size                   = UDim2.new(0.5, 0, 0.5, 0)
	frame.Position               = UDim2.new(offset.x > 0 and 0.5 or 0, 0, offset.y > 0 and 0.5 or 0, 0)
	frame.BackgroundTransparency = 1
	frame.BorderSizePixel        = 0
	frame.Parent                 = screenGui

	local editableImage = AssetService:CreateEditableImage({ Size = Vector2.new(TILE_W, TILE_H) })

	local imageLabel = Instance.new("ImageLabel")
	imageLabel.Size                   = UDim2.new(1, 0, 1, 0)
	imageLabel.BackgroundTransparency = 1
	imageLabel.ImageContent           = Content.fromObject(editableImage)
	imageLabel.ResampleMode           = Enum.ResamplerMode.Default
	imageLabel.Parent                 = frame

	local doneEvent = Instance.new("BindableEvent")
	doneEvent.Name   = "BindableEvent"
	doneEvent.Parent = frame

	local actorFolder = Instance.new("Folder")
	actorFolder.Name   = "ActorFolder"
	actorFolder.Parent = frame

	tiles[i] = {
		editableImage = editableImage,
		offsetX       = offset.x,
		offsetY       = offset.y,
		intOffsetX    = offset.ix,
		intOffsetY    = offset.iy,
		frame         = frame,
		doneEvent     = doneEvent,
		actorFolder   = actorFolder,
		accumBuffer   = buffer.create(TILE_W * TILE_H * 12),  -- HDR color accum (always full size, indexed at INT size when upscaling)
		skyBuffer     = buffer.create(TILE_W * TILE_H * 4),    -- sky mask: 1 byte per pixel (1=sky, 0=geometry)
		reconvBuffer  = buffer.create(TILE_W * TILE_H * 4),    -- reconstruction convergence counter per output pixel
		flushBuffer   = buffer.create(CHUNK_W * CHUNK_H * 4),
		displayBuffer = buffer.create(TILE_W * TILE_H * 4),           -- full-res output
		depthBuffer   = buffer.create(TILE_W * TILE_H * 4),   -- f32 hit distance
		normalBuffer  = buffer.create(TILE_W * TILE_H * 12),  -- f32 xyz normal
		varianceM2    = buffer.create(TILE_W * TILE_H * 4),   -- Welford M2 for variance
		bloomHalf     = buffer.create(BLOOM_W2 * BLOOM_H2 * 12),
		bloomQuarter  = buffer.create(BLOOM_W4 * BLOOM_H4 * 12),
		bloomTemp     = buffer.create(BLOOM_W2 * BLOOM_H2 * 12),
		frozenCount   = 0,       -- how many output pixels have reached sticky-frozen (G)
		touched       = false,   -- did any chunk update this tile's accum this pass (H)
	}
end

for i = 1, NUM_WORKERS do
	local tileIndex = math.ceil(i / (NUM_WORKERS / NUM_TILES))
	local tile      = tiles[tileIndex]
	local actor     = actorTemplate:Clone()
	actor.Parent    = tile.actorFolder
	workers[i]      = { actor = actor, tileIndex = tileIndex, busy = false }
end

-- ─── Lighting helpers ──────────────────────────────────────────────────────

local function gatherLights()
	local list = {}
	table.insert(list, {
		type      = "directional",
		direction = Vector3.new(-0.3, 0.85, 0.4).Unit,
		color     = Vector3.new(1.0, 0.95, 0.80),
		strength  = Lighting.Brightness * 0.6,
	})
	for _, obj in ipairs(workspace:GetDescendants()) do
		local part = obj.Parent
		if not (part and part:IsA("BasePart")) then continue end
		if obj:IsA("PointLight") then
			table.insert(list, {
				type = "point", position = part.Position,
				color = Vector3.new(obj.Color.R, obj.Color.G, obj.Color.B),
				brightness = obj.Brightness, range = obj.Range, sourcePart = part,
			})
		elseif obj:IsA("SpotLight") then
			table.insert(list, {
				type = "spot", position = part.Position,
				direction = part.CFrame:VectorToWorldSpace(Vector3.new(0, 0, -1)),
				color = Vector3.new(obj.Color.R, obj.Color.G, obj.Color.B),
				brightness = obj.Brightness, range = obj.Range,
				angle = math.rad(obj.Angle), sourcePart = part,
			})
		elseif obj:IsA("SurfaceLight") then
			table.insert(list, {
				type = "surface", position = part.Position,
				direction = part.CFrame:VectorToWorldSpace(Vector3.new(0, 0, -1)),
				color = Vector3.new(obj.Color.R, obj.Color.G, obj.Color.B),
				brightness = obj.Brightness, range = obj.Range,
				angle = math.rad(obj.Angle), sourcePart = part,
			})
		end
	end
	if #list > 21 then
		local camPos = player.Character and
			player.Character.HumanoidRootPart.Position or Vector3.zero
		table.sort(list, function(a, b)
			if a.type == "directional" then return true end
			if b.type == "directional" then return false end
			return (a.position - camPos).Magnitude < (b.position - camPos).Magnitude
		end)
		local capped, count = {}, 0
		for _, light in ipairs(list) do
			if light.type == "directional" or count < 20 then
				table.insert(capped, light)
				if light.type ~= "directional" then count += 1 end
			end
		end
		list = capped
	end
	return list
end

local function getSunDirection() return Lighting:GetSunDirection() end
local function getSunColor()
	local t = math.clamp((Lighting.ClockTime - 6) / 6, 0, 1)
	return Vector3.new(1.0, 0.70, 0.40):Lerp(Vector3.new(1.0, 0.97, 0.88), t)
end
local function getSkyBrightness()
	local clock = Lighting.ClockTime
	if clock >= 6 and clock <= 18 then return 1.0
	elseif clock > 18 and clock <= 20 then return 1.0 - ((clock - 18) / 2) * 0.95
	elseif clock > 4  and clock < 6  then return ((clock - 4) / 2) * 0.95 + 0.05
	else return 0.05 end
end

-- ─── Accum / flush ─────────────────────────────────────────────────────────

local function updateAccum(tile, colorBuf, depthBuf, normalBuf, chunkX, chunkY, chunkW, chunkH, passNum)
	local accum    = tile.accumBuffer
	local depth    = tile.depthBuffer
	local norm     = tile.normalBuffer
	local variance = tile.varianceM2
	local W        = UPSCALE_ENABLED and INT_TILE_W or TILE_W
	local n        = passNum
	tile.touched = true  -- this tile's accum changed this pass (H)
	for localY = 0, chunkH - 1 do
		local tileRow = (chunkY + localY) * W
		for localX = 0, chunkW - 1 do
			local tilePixel = tileRow + (chunkX + localX)
			local accumIdx  = tilePixel * 12
			local bufOffset = (localY * chunkW + localX) * 4

			-- Color: running average
			-- Alpha=0 means sky pixel, alpha=255 means geometry
			local isSky = buffer.readu8(colorBuf, bufOffset + 3) == 0
			local nr, ng, nb
			if isSky then
				-- Sky encoded as direct [0,1] values
				nr = buffer.readu8(colorBuf, bufOffset)     / 255
				ng = buffer.readu8(colorBuf, bufOffset + 1) / 255
				nb = buffer.readu8(colorBuf, bufOffset + 2) / 255
			else
				-- Geometry encoded with HDR scaling
				nr = buffer.readu8(colorBuf, bufOffset)     / 255 * HDR_MAX
				ng = buffer.readu8(colorBuf, bufOffset + 1) / 255 * HDR_MAX
				nb = buffer.readu8(colorBuf, bufOffset + 2) / 255 * HDR_MAX
			end
			-- Track sky mask for this pixel
			buffer.writeu8(tile.skyBuffer, tilePixel, isSky and 1 or 0)
			local ar = buffer.readf32(accum, accumIdx)
			local ag = buffer.readf32(accum, accumIdx + 4)
			local ab = buffer.readf32(accum, accumIdx + 8)
			local newR = (ar * (n-1) + nr) / n
			local newG = (ag * (n-1) + ng) / n
			local newB = (ab * (n-1) + nb) / n
			buffer.writef32(accum, accumIdx,     newR)
			buffer.writef32(accum, accumIdx + 4, newG)
			buffer.writef32(accum, accumIdx + 8, newB)

			-- Welford online variance (luminance)
			local lum    = nr * 0.2126 + ng * 0.7152 + nb * 0.0722
			local avgLum = newR * 0.2126 + newG * 0.7152 + newB * 0.0722
			local delta  = lum - avgLum
			local oldM2  = buffer.readf32(variance, tilePixel * 4)
			buffer.writef32(variance, tilePixel * 4, oldM2 + delta * delta)

			-- Depth: running average
			if depthBuf then
				local nd = buffer.readf32(depthBuf, (localY * chunkW + localX) * 4)
				local od = buffer.readf32(depth, tilePixel * 4)
				buffer.writef32(depth, tilePixel * 4, (od * (n-1) + nd) / n)
			end

			-- Normal: running average
			if normalBuf then
				local ni = (localY * chunkW + localX) * 12
				local nnx = buffer.readf32(normalBuf, ni)
				local nny = buffer.readf32(normalBuf, ni + 4)
				local nnz = buffer.readf32(normalBuf, ni + 8)
				local onx = buffer.readf32(norm, accumIdx)
				local ony = buffer.readf32(norm, accumIdx + 4)
				local onz = buffer.readf32(norm, accumIdx + 8)
				buffer.writef32(norm, accumIdx,     (onx * (n-1) + nnx) / n)
				buffer.writef32(norm, accumIdx + 4, (ony * (n-1) + nny) / n)
				buffer.writef32(norm, accumIdx + 8, (onz * (n-1) + nnz) / n)
			end
		end
	end
end

local function flushChunkToScreen(tile, chunkX, chunkY, chunkW, chunkH)
	local accum     = tile.accumBuffer
	local outBuffer = tile.flushBuffer
	for localY = 0, chunkH - 1 do
		local tileRow = (chunkY + localY) * TILE_W
		for localX = 0, chunkW - 1 do
			local tilePixel = tileRow + (chunkX + localX)
			local accumIdx  = tilePixel * 12
			local bufOffset = (localY * chunkW + localX) * 4
			local r = buffer.readf32(accum, accumIdx)
			local g = buffer.readf32(accum, accumIdx + 4)
			local b = buffer.readf32(accum, accumIdx + 8)
			local noise = buffer.readf32(DITHER_BUF, ((chunkY+localY)*TILE_W+(chunkX+localX)) * 4)
			buffer.writeu8(outBuffer, bufOffset,     dwrite(r, noise))
			buffer.writeu8(outBuffer, bufOffset + 1, dwrite(g, noise))
			buffer.writeu8(outBuffer, bufOffset + 2, dwrite(b, noise))
			buffer.writeu8(outBuffer, bufOffset + 3, 255)
		end
	end
	tile.editableImage:WritePixelsBuffer(
		Vector2.new(chunkX, chunkY), Vector2.new(chunkW, chunkH), outBuffer)
end

local function resetAccumBuffers()
	for _, tile in ipairs(tiles) do
		buffer.fill(tile.accumBuffer,     0, 0, TILE_W * TILE_H * 12)
		buffer.fill(tile.skyBuffer,       0, 0, TILE_W * TILE_H * 4)
		buffer.fill(tile.reconvBuffer,    0, 0, TILE_W * TILE_H * 4)
		buffer.fill(tile.depthBuffer,     0, 0, TILE_W * TILE_H * 4)
		buffer.fill(tile.normalBuffer,    0, 0, TILE_W * TILE_H * 12)
		buffer.fill(tile.varianceM2,      0, 0, TILE_W * TILE_H * 4)
		tile.frozenCount = 0
		tile.touched     = false
	end
end

-- ─── Edge-aware reconstruction upscaler ────────────────────────────────────
-- Upscales INT_TILE (480×270) → TILE (960×540) using depth+normal+variance
-- weighted 3×3 kernel. At low sample counts filters heavily; at high counts
-- reduces to near-native with slight sharpening.

-- Thresholds for hard depth/normal edge rejection
local DEPTH_REJECT_THRESHOLD  = 0.15  -- reject neighbor if relative depth diff exceeds this
local NORMAL_REJECT_THRESHOLD = 0.5   -- reject neighbor if dot product below this (cos ~60°)
local VARIANCE_SHARP_THRESHOLD = 0.02 -- sharpen only when pixel is this converged

local FROZEN_THRESHOLD = 3  -- consecutive stable reconstruction passes before pixel is frozen

local function reconstructTile(tile, passNum)
	local accum    = tile.accumBuffer
	local depth    = tile.depthBuffer
	local norm     = tile.normalBuffer
	local variance = tile.varianceM2
	local display  = tile.displayBuffer
	local reconvBuf = tile.reconvBuffer
	local skyB     = tile.skyBuffer
	local IW       = INT_TILE_W
	local IH       = INT_TILE_H
	local OW       = TILE_W
	local OH       = TILE_H

	-- Filter strength fades as samples accumulate
	local filterStrength = math.max(0.05, 1.0 - passNum / 40.0)
	local invPass        = 1 / math.max(1, passNum)

	for oy = 0, OH - 1 do
		for ox = 0, OW - 1 do
			local ri = oy * OW + ox
			local rc = buffer.readu8(reconvBuf, ri)

			-- Sticky-frozen pixel (B): once a pixel has held converged for
			-- FROZEN_THRESHOLD reconstructions it is skipped for good. Its last
			-- written display value (a bilinear result, see below) is held. With
			-- a locked camera and monotonically-decreasing variance this can't go
			-- stale, so freezing is permanent rather than reset every pass.
			if rc >= FROZEN_THRESHOLD then
				-- held; no work
			else
				-- Fractional pixel-center mapping: works at any scale ratio
				local fsx = (ox + 0.5) * IW / OW - 0.5
				local fsy = (oy + 0.5) * IH / OH - 0.5
				local ix0 = math.max(0, math.min(math.floor(fsx), IW - 1))
				local iy0 = math.max(0, math.min(math.floor(fsy), IH - 1))

				local centerIdx = iy0 * IW + ix0
				local ci0  = centerIdx * 12
				local cdi0 = centerIdx * 4
				local cvar = buffer.readf32(variance, cdi0) * invPass
				local cr0  = buffer.readf32(accum, ci0)
				local cg0  = buffer.readf32(accum, ci0 + 4)
				local cb0  = buffer.readf32(accum, ci0 + 8)
				local centerSky = buffer.readu8(skyB, centerIdx) == 1

				local di      = ri * 4
				local noise   = buffer.readf32(DITHER_BUF, di)   -- D: precomputed dither
				local writeFn = centerSky and dwriteSky or dwrite

				if cvar < VARIANCE_SHARP_THRESHOLD then
					-- Converged (C): depth/sky-guarded BILINEAR of the four internal
					-- texels around the fractional source position — replaces the old
					-- nearest-neighbour blit that produced hard 2x2 blocks on smooth
					-- gradients. Taps of a different sky-ness, or with >15% relative
					-- depth difference, are dropped so silhouettes/horizon stay crisp.
					local cd  = buffer.readf32(depth, cdi0)
					local tx  = fsx - math.floor(fsx); if tx < 0 then tx = 0 elseif tx > 1 then tx = 1 end
					local ty  = fsy - math.floor(fsy); if ty < 0 then ty = 0 elseif ty > 1 then ty = 1 end
					local ix1 = math.min(ix0 + 1, IW - 1)
					local iy1 = math.min(iy0 + 1, IH - 1)
					local dTol = 0.15 * math.max(cd, 0.001)

					local sumR, sumG, sumB, sumW = 0, 0, 0, 0

					-- tap (ix0,iy0) — the center texel, always admissible
					do
						local w = (1 - tx) * (1 - ty)
						local si = centerIdx * 12
						sumR += cr0 * w; sumG += cg0 * w; sumB += cb0 * w; sumW += w
					end
					-- tap (ix1,iy0)
					do
						local w = tx * (1 - ty)
						if w > 0 then
							local sidx = iy0 * IW + ix1
							if (buffer.readu8(skyB, sidx) == 1) == centerSky
								and math.abs(buffer.readf32(depth, sidx * 4) - cd) <= dTol then
								local si = sidx * 12
								sumR += buffer.readf32(accum, si) * w
								sumG += buffer.readf32(accum, si + 4) * w
								sumB += buffer.readf32(accum, si + 8) * w
								sumW += w
							end
						end
					end
					-- tap (ix0,iy1)
					do
						local w = (1 - tx) * ty
						if w > 0 then
							local sidx = iy1 * IW + ix0
							if (buffer.readu8(skyB, sidx) == 1) == centerSky
								and math.abs(buffer.readf32(depth, sidx * 4) - cd) <= dTol then
								local si = sidx * 12
								sumR += buffer.readf32(accum, si) * w
								sumG += buffer.readf32(accum, si + 4) * w
								sumB += buffer.readf32(accum, si + 8) * w
								sumW += w
							end
						end
					end
					-- tap (ix1,iy1)
					do
						local w = tx * ty
						if w > 0 then
							local sidx = iy1 * IW + ix1
							if (buffer.readu8(skyB, sidx) == 1) == centerSky
								and math.abs(buffer.readf32(depth, sidx * 4) - cd) <= dTol then
								local si = sidx * 12
								sumR += buffer.readf32(accum, si) * w
								sumG += buffer.readf32(accum, si + 4) * w
								sumB += buffer.readf32(accum, si + 8) * w
								sumW += w
							end
						end
					end

					local fr, fg, fb
					if sumW > 0 then
						fr = sumR / sumW; fg = sumG / sumW; fb = sumB / sumW
					else
						fr = cr0; fg = cg0; fb = cb0
					end

					buffer.writeu8(display, di,     writeFn(fr, noise))
					buffer.writeu8(display, di + 1, writeFn(fg, noise))
					buffer.writeu8(display, di + 2, writeFn(fb, noise))
					buffer.writeu8(display, di + 3, 255)

					-- Advance toward frozen; tally the tile when this pixel first freezes (G)
					rc = rc + 1
					buffer.writeu8(reconvBuf, ri, rc)
					if rc == FROZEN_THRESHOLD then
						tile.frozenCount += 1
					end
				else
					-- Not converged: full edge-aware kernel (math unchanged). Hysteresis
					-- (B): a single high-variance pass costs one freeze credit instead of
					-- resetting to zero, so pixels hovering near the threshold still
					-- eventually freeze instead of paying kernel cost forever.
					local cd   = buffer.readf32(depth, cdi0)
					local cnx  = buffer.readf32(norm, ci0)
					local cny  = buffer.readf32(norm, ci0 + 4)
					local cnz  = buffer.readf32(norm, ci0 + 8)

					-- Adaptive kernel radius: 3x3 for medium/high variance, 1x1 for low
					local radius = math.min(2, math.max(0, math.ceil(cvar * 20 * filterStrength)))
					-- Cap at 1 after pass 30 — variance is low everywhere, avoid overhead
					if passNum > 30 then radius = math.min(radius, 1) end
					-- Clamp at tile borders
					if ix0 < radius or iy0 < radius or ix0 >= IW - radius or iy0 >= IH - radius then
						radius = 0
					end

					local sumR, sumG, sumB, sumW = 0, 0, 0, 0

					for ky = -radius, radius do
						for kx = -radius, radius do
							local ix = math.max(0, math.min(ix0 + kx, IW - 1))
							local iy = math.max(0, math.min(iy0 + ky, IH - 1))
							local si  = (iy * IW + ix) * 12
							local sdi = (iy * IW + ix) * 4

							-- Hard depth edge rejection: skip neighbor if depth differs too much
							local sd = buffer.readf32(depth, sdi)
							local depthRel = math.abs(sd - cd) / math.max(cd, 0.001)
							if depthRel > DEPTH_REJECT_THRESHOLD then
								-- depth edge — skip this neighbor entirely
							else
								-- Hard normal edge rejection: skip neighbor if surfaces face different directions
								local snx = buffer.readf32(norm, si)
								local sny = buffer.readf32(norm, si + 4)
								local snz = buffer.readf32(norm, si + 8)
								local normalDot = cnx*snx + cny*sny + cnz*snz
								if normalDot < NORMAL_REJECT_THRESHOLD then
									-- normal edge — skip this neighbor entirely
								else
									-- Spatial weight (Gaussian, fractional distance)
									local dx = ix - fsx
									local dy = iy - fsy
									local wSpatial = math.exp(-(dx*dx + dy*dy) * 0.5)

									-- Color similarity weight
									local sr2 = buffer.readf32(accum, si)
									local sg2 = buffer.readf32(accum, si + 4)
									local sb2 = buffer.readf32(accum, si + 8)
									local colorDiff = (sr2-cr0)^2 + (sg2-cg0)^2 + (sb2-cb0)^2
									local wColor = math.exp(-colorDiff * 8.0)

									local w = wSpatial * wColor * filterStrength
										+ (1 - filterStrength) * (kx == 0 and ky == 0 and 1 or 0)

									sumR += sr2 * w
									sumG += sg2 * w
									sumB += sb2 * w
									sumW += w
								end
							end
						end
					end

					local fr = sumW > 0 and sumR / sumW or cr0
					local fg = sumW > 0 and sumG / sumW or cg0
					local fb = sumW > 0 and sumB / sumW or cb0

					buffer.writeu8(display, di,     writeFn(fr, noise))
					buffer.writeu8(display, di + 1, writeFn(fg, noise))
					buffer.writeu8(display, di + 2, writeFn(fb, noise))
					buffer.writeu8(display, di + 3, 255)

					-- Hysteresis down-count (B): lose one credit, don't reset
					if rc > 0 then buffer.writeu8(reconvBuf, ri, rc - 1) end
				end
			end
		end
	end

	tile.editableImage:WritePixelsBuffer(Vector2.new(0, 0), Vector2.new(OW, OH), display)
end

-- Reconstruct a tile only when it can change the image (G, H). A fully-frozen
-- tile's pixels are all held, and an untouched tile (no chunks this pass) has an
-- unchanged accum buffer; past pass 40 the filter strength is floored so an
-- untouched tile would reconstruct bit-identically. Either way, skip the whole
-- main-thread pass — this is what trims the SLOW HANDLER stall at high pass counts.
local RECONSTRUCT_TOTAL = TILE_W * TILE_H
local function maybeReconstruct(tile, passNum)
	if tile.frozenCount >= RECONSTRUCT_TOTAL then return end          -- G
	if (not tile.touched) and passNum > 40 then return end            -- H
	reconstructTile(tile, passNum)
end

-- ─── Bloom ─────────────────────────────────────────────────────────────────

local function gaussBlurH(src, dst, W, H)
	for y = 0, H - 1 do
		for x = 0, W - 1 do
			local dstIdx = (y*W+x)*12
			local sr, sg, sb = 0, 0, 0
			for k = 1, 5 do
				local sx = math.clamp(x+k-3, 0, W-1)
				local si = (y*W+sx)*12
				local w  = GAUSS5[k]
				sr += buffer.readf32(src, si)   * w
				sg += buffer.readf32(src, si+4) * w
				sb += buffer.readf32(src, si+8) * w
			end
			buffer.writef32(dst, dstIdx,   sr)
			buffer.writef32(dst, dstIdx+4, sg)
			buffer.writef32(dst, dstIdx+8, sb)
		end
	end
end

local function gaussBlurV(src, dst, W, H)
	for y = 0, H - 1 do
		for x = 0, W - 1 do
			local dstIdx = (y*W+x)*12
			local sr, sg, sb = 0, 0, 0
			for k = 1, 5 do
				local sy = math.clamp(y+k-3, 0, H-1)
				local si = (sy*W+x)*12
				local w  = GAUSS5[k]
				sr += buffer.readf32(src, si)   * w
				sg += buffer.readf32(src, si+4) * w
				sb += buffer.readf32(src, si+8) * w
			end
			buffer.writef32(dst, dstIdx,   sr)
			buffer.writef32(dst, dstIdx+4, sg)
			buffer.writef32(dst, dstIdx+8, sb)
		end
	end
end

local function computeBloom(tile)
	local accum  = tile.accumBuffer
	local half   = tile.bloomHalf
	local qtr    = tile.bloomQuarter
	local temp   = tile.bloomTemp
	local thresh = settings.bloomThreshold / settings.exposure

	for y = 0, BLOOM_H2-1 do
		for x = 0, BLOOM_W2-1 do
			local di = (y*BLOOM_W2+x)*12
			local x0=x*2; local x1=math.min(x0+1,TILE_W-1)
			local y0=y*2; local y1=math.min(y0+1,TILE_H-1)
			local i00=(y0*TILE_W+x0)*12; local i10=(y0*TILE_W+x1)*12
			local i01=(y1*TILE_W+x0)*12; local i11=(y1*TILE_W+x1)*12
			local r=(buffer.readf32(accum,i00)  +buffer.readf32(accum,i10)  +buffer.readf32(accum,i01)  +buffer.readf32(accum,i11))  *0.25
			local g=(buffer.readf32(accum,i00+4)+buffer.readf32(accum,i10+4)+buffer.readf32(accum,i01+4)+buffer.readf32(accum,i11+4))*0.25
			local b=(buffer.readf32(accum,i00+8)+buffer.readf32(accum,i10+8)+buffer.readf32(accum,i01+8)+buffer.readf32(accum,i11+8))*0.25
			local lum = r*0.2126+g*0.7152+b*0.0722
			local fac = math.max(0, lum-thresh)/(lum+0.001)
			buffer.writef32(half,di,  r*fac); buffer.writef32(half,di+4,g*fac); buffer.writef32(half,di+8,b*fac)
		end
	end

	gaussBlurH(half,temp,BLOOM_W2,BLOOM_H2); gaussBlurV(temp,half,BLOOM_W2,BLOOM_H2)
	gaussBlurH(half,temp,BLOOM_W2,BLOOM_H2); gaussBlurV(temp,half,BLOOM_W2,BLOOM_H2)

	for y = 0, BLOOM_H4-1 do
		for x = 0, BLOOM_W4-1 do
			local di=(y*BLOOM_W4+x)*12
			local x0=x*2; local x1=math.min(x0+1,BLOOM_W2-1)
			local y0=y*2; local y1=math.min(y0+1,BLOOM_H2-1)
			local i00=(y0*BLOOM_W2+x0)*12; local i10=(y0*BLOOM_W2+x1)*12
			local i01=(y1*BLOOM_W2+x0)*12; local i11=(y1*BLOOM_W2+x1)*12
			buffer.writef32(qtr,di,  (buffer.readf32(half,i00)  +buffer.readf32(half,i10)  +buffer.readf32(half,i01)  +buffer.readf32(half,i11))  *0.25)
			buffer.writef32(qtr,di+4,(buffer.readf32(half,i00+4)+buffer.readf32(half,i10+4)+buffer.readf32(half,i01+4)+buffer.readf32(half,i11+4))*0.25)
			buffer.writef32(qtr,di+8,(buffer.readf32(half,i00+8)+buffer.readf32(half,i10+8)+buffer.readf32(half,i01+8)+buffer.readf32(half,i11+8))*0.25)
		end
	end

	gaussBlurH(qtr,temp,BLOOM_W4,BLOOM_H4); gaussBlurV(temp,qtr,BLOOM_W4,BLOOM_H4)
	gaussBlurH(qtr,temp,BLOOM_W4,BLOOM_H4); gaussBlurV(temp,qtr,BLOOM_W4,BLOOM_H4)

	for y = 0, BLOOM_H2-1 do
		for x = 0, BLOOM_W2-1 do
			local fx=math.max(0,x*0.5-0.25); local fy=math.max(0,y*0.5-0.25)
			local x0=math.min(math.floor(fx),BLOOM_W4-1); local x1=math.min(x0+1,BLOOM_W4-1)
			local y0=math.min(math.floor(fy),BLOOM_H4-1); local y1=math.min(y0+1,BLOOM_H4-1)
			local tx=fx-math.floor(fx); local itx=1-tx
			local ty=fy-math.floor(fy); local ity=1-ty
			local q00=(y0*BLOOM_W4+x0)*12; local q10=(y0*BLOOM_W4+x1)*12
			local q01=(y1*BLOOM_W4+x0)*12; local q11=(y1*BLOOM_W4+x1)*12
			local hi=(y*BLOOM_W2+x)*12
			buffer.writef32(half,hi,  buffer.readf32(half,hi)  +buffer.readf32(qtr,q00)*itx*ity+buffer.readf32(qtr,q10)*tx*ity+buffer.readf32(qtr,q01)*itx*ty+buffer.readf32(qtr,q11)*tx*ty)
			buffer.writef32(half,hi+4,buffer.readf32(half,hi+4)+buffer.readf32(qtr,q00+4)*itx*ity+buffer.readf32(qtr,q10+4)*tx*ity+buffer.readf32(qtr,q01+4)*itx*ty+buffer.readf32(qtr,q11+4)*tx*ty)
			buffer.writef32(half,hi+8,buffer.readf32(half,hi+8)+buffer.readf32(qtr,q00+8)*itx*ity+buffer.readf32(qtr,q10+8)*tx*ity+buffer.readf32(qtr,q01+8)*itx*ty+buffer.readf32(qtr,q11+8)*tx*ty)
		end
	end
end

local function flushTileWithBloom(tile)
	local accum=tile.accumBuffer; local half=tile.bloomHalf; local displayBuf=tile.displayBuffer
	local W=TILE_W; local H=TILE_H; local W2=BLOOM_W2; local bi=settings.bloomIntensity
	for y = 0, H-1 do
		local y0=BLOOM_UY0[y]; local y1=BLOOM_UY1[y]; local ty=BLOOM_UTY[y]; local ity=1-ty
		for x = 0, W-1 do
			local x0=BLOOM_UX0[x]; local x1=BLOOM_UX1[x]; local tx=BLOOM_UTX[x]; local itx=1-tx
			local h00=(y0*W2+x0)*12; local h10=(y0*W2+x1)*12
			local h01=(y1*W2+x0)*12; local h11=(y1*W2+x1)*12
			local w00=itx*ity; local w10=tx*ity; local w01=itx*ty; local w11=tx*ty
			local br=buffer.readf32(half,h00)*w00+buffer.readf32(half,h10)*w10+buffer.readf32(half,h01)*w01+buffer.readf32(half,h11)*w11
			local bg=buffer.readf32(half,h00+4)*w00+buffer.readf32(half,h10+4)*w10+buffer.readf32(half,h01+4)*w01+buffer.readf32(half,h11+4)*w11
			local bb=buffer.readf32(half,h00+8)*w00+buffer.readf32(half,h10+8)*w10+buffer.readf32(half,h01+8)*w01+buffer.readf32(half,h11+8)*w11
			local ai=(y*W+x)*12
			local r=buffer.readf32(accum,ai)+br*bi
			local g=buffer.readf32(accum,ai+4)+bg*bi
			local b=buffer.readf32(accum,ai+8)+bb*bi
			local di=(y*W+x)*4
			local noise=buffer.readf32(DITHER_BUF, di)
			buffer.writeu8(displayBuf,di,  dwrite(r,noise))
			buffer.writeu8(displayBuf,di+1,dwrite(g,noise))
			buffer.writeu8(displayBuf,di+2,dwrite(b,noise))
			buffer.writeu8(displayBuf,di+3,255)
		end
	end
	tile.editableImage:WritePixelsBuffer(Vector2.new(0,0),Vector2.new(TILE_W,TILE_H),displayBuf)
end

local function flushTileNoBloom(tile)
	local accum=tile.accumBuffer; local displayBuf=tile.displayBuffer; local sky=tile.skyBuffer
	local W=TILE_W; local H=TILE_H
	for y = 0, H-1 do
		for x = 0, W-1 do
			local ai=(y*W+x)*12
			local r=buffer.readf32(accum,ai)
			local g=buffer.readf32(accum,ai+4)
			local b=buffer.readf32(accum,ai+8)
			local di=(y*W+x)*4
			local noise=buffer.readf32(DITHER_BUF, di)
			local writeFn = buffer.readu8(sky, y*W+x) == 1 and dwriteSky or dwrite
			buffer.writeu8(displayBuf,di,  writeFn(r,noise))
			buffer.writeu8(displayBuf,di+1,writeFn(g,noise))
			buffer.writeu8(displayBuf,di+2,writeFn(b,noise))
			buffer.writeu8(displayBuf,di+3,255)
		end
	end
	tile.editableImage:WritePixelsBuffer(Vector2.new(0,0),Vector2.new(TILE_W,TILE_H),displayBuf)
end

-- ─── Render loop ───────────────────────────────────────────────────────────

local isRendering    = false
local microTileQueue = {}
local nextTileIndex  = 1
local chunksComplete = 0
local totalChunks    = 0
local passNumber     = 0
local startTime      = 0

-- Adaptive supersampling: skip chunks whose pixels haven't changed meaningfully
-- since the last snapshot. This correctly handles reflections and other
-- high-frequency content — convergence is detected by stability, not variance.
local AS_CHANGE_THRESHOLD = 0.02   -- relative variance below this = converged
local AS_MIN_PASSES       = 20     -- don't skip before this pass
local AS_SAMPLE_STRIDE    = 6      -- check every Nth pixel in chunk

-- Relative variance convergence check.
-- Uses coefficient of variation: M2 / (passNum * mean²)
-- This correctly handles reflections — a bright stable reflection has
-- low relative variance even though absolute variance may be high.
-- Sample average normalized variance for a chunk (used for adaptive AO)
local function sampleChunkVariance(tile, chunkX, chunkY, chunkW, chunkH, passNum)
	if passNum < 2 then return 1.0 end
	local variance = tile.varianceM2
	local accum    = tile.accumBuffer
	local W        = UPSCALE_ENABLED and INT_TILE_W or TILE_W
	local sum, count = 0, 0
	for ly = 0, chunkH - 1, AS_SAMPLE_STRIDE do
		for lx = 0, chunkW - 1, AS_SAMPLE_STRIDE do
			local ai = ((chunkY + ly) * W + (chunkX + lx)) * 12
			local vi = ((chunkY + ly) * W + (chunkX + lx)) * 4
			local r = buffer.readf32(accum, ai)
			local g = buffer.readf32(accum, ai + 4)
			local b = buffer.readf32(accum, ai + 8)
			local mean = r * 0.2126 + g * 0.7152 + b * 0.0722
			local m2   = buffer.readf32(variance, vi)
			sum += m2 / (passNum * math.max(mean * mean, 0.0001))
			count += 1
		end
	end
	return count > 0 and (sum / count) or 1.0
end

local function isChunkConverged(tile, chunkX, chunkY, chunkW, chunkH, passNum)
	if passNum < AS_MIN_PASSES then return false end
	local accum    = tile.accumBuffer
	local variance = tile.varianceM2
	local depth    = tile.depthBuffer
	local W        = UPSCALE_ENABLED and INT_TILE_W or TILE_W
	local sumRelVar = 0
	local count     = 0
	for ly = 0, chunkH - 1, AS_SAMPLE_STRIDE do
		for lx = 0, chunkW - 1, AS_SAMPLE_STRIDE do
			local px = chunkX + lx
			local py = chunkY + ly
			local ai = (py * W + px) * 12
			local vi = (py * W + px) * 4
			local di = (py * W + px) * 4

			-- Skip edge pixels: check if neighboring pixel has significantly different depth
			-- Edges are always high variance and shouldn't disqualify the whole chunk
			local isEdge = false
			if px > 0 and px < W - 1 and py > 0 then
				local cd = buffer.readf32(depth, di)
				if cd > 0.001 then
					local nd = buffer.readf32(depth, (py * W + px + 1) * 4)
					if math.abs(nd - cd) / cd > 0.15 then isEdge = true end
					if not isEdge then
						nd = buffer.readf32(depth, (py * W + px - 1) * 4)
						if math.abs(nd - cd) / cd > 0.15 then isEdge = true end
					end
				end
			end

			if not isEdge then
				local r = buffer.readf32(accum, ai)
				local g = buffer.readf32(accum, ai + 4)
				local b = buffer.readf32(accum, ai + 8)
				local mean = r * 0.2126 + g * 0.7152 + b * 0.0722
				local m2   = buffer.readf32(variance, vi)
				local relVar = m2 / (passNum * math.max(mean * mean, 0.0001))
				sumRelVar += relVar
				count += 1
			end
		end
	end
	if count == 0 then return true end  -- all edge pixels, consider converged
	local threshold = AS_CHANGE_THRESHOLD / math.sqrt(passNum / AS_MIN_PASSES)
	return (sumRelVar / count) < threshold
end

local function buildQueue()
	microTileQueue = {}
	local cw = CHUNK_W
	local ch = CHUNK_H
	-- Render at internal (half) resolution
	local tileW = UPSCALE_ENABLED and INT_TILE_W or TILE_W
	local tileH = UPSCALE_ENABLED and INT_TILE_H or TILE_H
	for tileIndex, tile in ipairs(tiles) do
		local cols = math.ceil(tileW / cw)
		local rows = math.ceil(tileH / ch)
		for row = 0, rows-1 do
			for col = 0, cols-1 do
				local chunkX = col * cw
				local chunkY = row * ch
				local chunkW = math.min(cw, tileW - chunkX)
				local chunkH = math.min(ch, tileH - chunkY)
				-- Adaptive supersampling: skip converged chunks
				if not isChunkConverged(tile, chunkX, chunkY, chunkW, chunkH, passNumber) then
					table.insert(microTileQueue, {
						chunkX      = chunkX,
						chunkY      = chunkY,
						chunkW      = chunkW,
						chunkH      = chunkH,
						tileIndex   = tileIndex,
						tileOffsetX = UPSCALE_ENABLED and tile.intOffsetX or tile.offsetX,
						tileOffsetY = UPSCALE_ENABLED and tile.intOffsetY or tile.offsetY,
					})
				end
			end
		end
	end
	totalChunks = #microTileQueue
end

local function assignNextChunk(workerIndex)
	if not isRendering then workers[workerIndex].busy = false; return end
	if nextTileIndex > #microTileQueue then workers[workerIndex].busy = false; return end
	local chunk = microTileQueue[nextTileIndex]
	nextTileIndex += 1
	workers[workerIndex].busy = true
	workers[workerIndex].actor:SendMessage("RenderChunk", {
		chunkX          = chunk.chunkX,
		chunkY          = chunk.chunkY,
		chunkW          = chunk.chunkW,
		chunkH          = chunk.chunkH,
		tileIndex       = chunk.tileIndex,
		tileOffsetX     = chunk.tileOffsetX,
		tileOffsetY     = chunk.tileOffsetY,
		workerIndex     = workerIndex,
		FULL_W          = UPSCALE_ENABLED and INT_FULL_W or FULL_W,
		FULL_H          = UPSCALE_ENABLED and INT_FULL_H or FULL_H,
		camPos          = renderData.camPos,
		camRight        = renderData.camRight,
		camUp           = renderData.camUp,
		camLook         = renderData.camLook,
		halfWidth       = renderData.halfWidth,
		halfHeight      = renderData.halfHeight,
		lights          = renderData.lights,
		sunStrength     = renderData.sunStrength,
		passNumber      = passNumber,
		seed            = (passNumber % 100) * 100 + workerIndex,
		sunDirection    = renderData.sunDirection,
		sunColor        = renderData.sunColor,
		skyBrightness   = renderData.skyBrightness,
		characterFilter = renderData.characterFilter,
		bounceStrength  = settings.bounceStrength,
		sunSampleProb   = settings.sunSampleProb,
		aoRays          = settings.aoRays,
		maxBounces      = settings.maxBounces,
		chunkVariance   = sampleChunkVariance(tiles[chunk.tileIndex], chunk.chunkX, chunk.chunkY, chunk.chunkW, chunk.chunkH, passNumber),
		-- Subpixel jitter for ASR: Halton(passNumber, 2/3) in internal pixel units,
		-- re-centered to [-0.5, 0.5) so its MEAN is 0. The raw Halton sequence is
		-- [0,1) with mean +0.5, which biased every pass a half-internal-pixel toward
		-- +x/+y — a persistent sub-pixel smear vs where reconstruction samples. The
		-- worker adds this to (globalX + s1 - 0.5), which is already zero-mean.
		asrJitterX     = UPSCALE_ENABLED and (halton(passNumber, 2) - 0.5) or 0,
		asrJitterY     = UPSCALE_ENABLED and (halton(passNumber, 3) - 0.5) or 0,
	})
end

local function startNewPass()
	passNumber    += 1
	nextTileIndex  = 1
	chunksComplete = 0
	for _, dt in ipairs(tiles) do dt.touched = false end  -- reset per-pass touch flag (H)
	buildQueue()
	passLabel.Text = "Pass " .. passNumber .. (totalChunks < 10 and " (converging)" or "")
	-- If all chunks converged and queue is empty, run reconstruction and start next pass immediately
	if totalChunks == 0 then
		if UPSCALE_ENABLED and passNumber % 4 == 0 then
			for _, dt in ipairs(tiles) do maybeReconstruct(dt, passNumber) end
		end
		startNewPass()
		return
	end
	for wi = 1, NUM_WORKERS do
		workers[wi].busy = false
		assignNextChunk(wi)
	end
end

local reconstructsPending = 0

-- Finish the current pass: log, then start the next. Called directly when no
-- reconstruction is needed, or by the reconstruction barrier once all tiles report back.

local function finishPass()
	local elapsed = os.clock() - startTime
	print(string.format("Pass %d complete: %.2fs (chunks: %d)", passNumber, elapsed, totalChunks))
	startTime = os.clock()
	startNewPass()
end

-- Hand ASR reconstruction to worker Actors (one worker per tile) so the kernel runs
-- across cores in parallel instead of serially on the Controller's main thread. The
-- persistent display buffer is sent along so frozen pixels keep their last value.
-- Same G/H skip as maybeReconstruct. Results return as kind="reconstruct" events;
-- when every dispatched tile is back, finishPass() advances. If nothing needs work,
-- advances immediately.
local function dispatchParallelReconstruct()
	reconstructsPending = 0
	for ti = 1, NUM_TILES do
		local tile = tiles[ti]
		if tile.frozenCount >= RECONSTRUCT_TOTAL then
			-- G: fully frozen — display already correct, skip
		elseif (not tile.touched) and passNumber > 40 then
			-- H: untouched and filter strength floored — output identical, skip
		else
			reconstructsPending += 1
			-- tile ti's own first worker lives under tile ti's ActorFolder, so its
			-- BindableEvent fire routes back to tile ti's doneEvent naturally.
			workers[ti * 2 - 1].actor:SendMessage("ReconstructTile", {
				kind         = "reconstruct",
				tileIndex    = ti,
				passNumber   = passNumber,
				exposure     = settings.exposure,
				TILE_W       = TILE_W,
				TILE_H       = TILE_H,
				INT_W        = INT_TILE_W,
				INT_H        = INT_TILE_H,
				accumBuffer  = tile.accumBuffer,
				depthBuffer  = tile.depthBuffer,
				normalBuffer = tile.normalBuffer,
				varianceM2   = tile.varianceM2,
				skyBuffer    = tile.skyBuffer,
				reconvBuffer = tile.reconvBuffer,
				displayBuffer= tile.displayBuffer,
			})
		end
	end
	if reconstructsPending == 0 then
		finishPass()
	end
end

-- ─── Realtime mode state & display ──────────────────────────────────────────
-- Stepped resolution presets (16:9; each tile is half of full in both axes, and the
-- tile height is even so it splits cleanly into two worker halves).
local RT_PRESETS = {
	{ w = 128, h = 72  },   -- 1  lowest  (fastest, very blocky)
	{ w = 192, h = 108 },   -- 2
	{ w = 256, h = 144 },   -- 3
	{ w = 320, h = 180 },   -- 4  default
	{ w = 384, h = 216 },   -- 5
	{ w = 448, h = 252 },   -- 6  highest (slowest)
}

local RT_FULL_W, RT_FULL_H = 320, 180
local RT_TILE_W, RT_TILE_H = 160, 90
local rtTileOffsets = {}
local realtimeActive  = false
local realtimeBusy    = false
local realtimePending = 0
local rtLights        = nil   -- cached lights (don't rescan workspace every frame)
local rtImages        = {}
local rtAppliedIndex  = 0
local rtLastFrame     = 0     -- os.clock() at last completed frame (for FPS)
local rtFps           = 0     -- smoothed frames/sec
-- Temporal accumulation (reset-on-move): converges a still view, hard-resets when the
-- camera moves so no stale data smears (unlike an EMA blend).
local rtFrameN        = 0     -- frames accumulated at the current camera pose
local rtFrameCounter  = 0     -- ever-incrementing, varies the sample seed each frame
local rtLastCamCF     = nil   -- camera CFrame last frame (move detection)
local RT_MAX_ACCUM    = 512   -- cap so the running mean keeps adapting slightly
local rtAppliedPixel  = true  -- last-applied value of settings.rtPixelated (matches default)
local rtAppliedWindow = 0     -- last-applied value of settings.rtWindowMode
local RT_WIN_W        = 640   -- standardized "windowed" width (height follows aspect)

-- Dim backdrop behind the realtime window (windowed / 1:1 modes) so it reads as a clean
-- viewer instead of floating over the live 3D view.
local rtBackdrop = Instance.new("Frame")
rtBackdrop.Size                   = UDim2.new(1, 0, 1, 0)
rtBackdrop.Position               = UDim2.new(0, 0, 0, 0)
rtBackdrop.BackgroundColor3       = Color3.fromRGB(0, 0, 0)
rtBackdrop.BackgroundTransparency = 0.15
rtBackdrop.BorderSizePixel        = 0
rtBackdrop.ZIndex                 = 1
rtBackdrop.Visible                = false
rtBackdrop.Parent                 = screenGui

-- Position/size the four realtime tile images for the current window mode:
--   1 = fullscreen (2x2 quadrants filling the screen)
--   2 = windowed   (fixed ~640-wide centred window)
--   3 = 1:1        (native pixels, centred — sharpest, no scaling)
local function applyRealtimeLayout()
	local mode = settings.rtWindowMode or 1
	rtBackdrop.Visible = realtimeActive and mode ~= 1
	for i = 1, NUM_TILES do
		local lbl = rtImages[i] and rtImages[i].label
		if lbl then
			local rto = rtTileOffsets[i]
			local col = (rto.x > 0) and 1 or 0
			local row = (rto.y > 0) and 1 or 0
			if mode == 3 then
				-- 1:1 native, centred cluster
				lbl.Size     = UDim2.new(0, RT_TILE_W, 0, RT_TILE_H)
				lbl.Position = UDim2.new(0.5, -RT_FULL_W / 2 + col * RT_TILE_W, 0.5, -RT_FULL_H / 2 + row * RT_TILE_H)
			elseif mode == 2 then
				-- standardized windowed, centred, aspect-preserved
				local winW = RT_WIN_W
				local winH = winW * RT_FULL_H / RT_FULL_W
				local tw, th = winW / 2, winH / 2
				lbl.Size     = UDim2.new(0, tw, 0, th)
				lbl.Position = UDim2.new(0.5, -winW / 2 + col * tw, 0.5, -winH / 2 + row * th)
			else
				-- fullscreen quadrants
				lbl.Size     = UDim2.new(0.5, 0, 0.5, 0)
				lbl.Position = UDim2.new(0.5 * col, 0, 0.5 * row, 0)
			end
		end
	end
	rtAppliedWindow = mode
end

-- (Re)build the realtime display images at the current preset. EditableImages can't be
-- resized, so a resolution change destroys and recreates them. Cheap and only happens
-- when the slider step changes.
local function applyRealtimeResolution(idx)
	idx = math.clamp(idx, 1, #RT_PRESETS)
	local p = RT_PRESETS[idx]
	RT_FULL_W, RT_FULL_H = p.w, p.h
	RT_TILE_W, RT_TILE_H = p.w // 2, p.h // 2
	rtTileOffsets = {
		{ x = 0,         y = 0         },
		{ x = RT_TILE_W, y = 0         },
		{ x = 0,         y = RT_TILE_H },
		{ x = RT_TILE_W, y = RT_TILE_H },
	}
	for i = 1, NUM_TILES do
		local old = rtImages[i]
		if old then
			if old.label then old.label:Destroy() end
			if old.image then old.image:Destroy() end
		end
		local img = AssetService:CreateEditableImage({ Size = Vector2.new(RT_TILE_W, RT_TILE_H) })
		local lbl = Instance.new("ImageLabel")
		lbl.Size                   = UDim2.new(0.5, 0, 0.5, 0)
		lbl.BackgroundTransparency = 1
		lbl.ImageContent           = Content.fromObject(img)
		lbl.ResampleMode           = settings.rtPixelated and Enum.ResamplerMode.Pixelated or Enum.ResamplerMode.Default
		lbl.Visible                = realtimeActive
		lbl.ZIndex                 = 2
		lbl.Parent                 = screenGui   -- fullscreen layer, positioned by applyRealtimeLayout
		rtImages[i] = { image = img, label = lbl }
	end
	rtAppliedIndex = idx
	rtFrameN       = 0
	rtLastCamCF    = nil   -- forces a reset next frame; worker reallocates its accum on size change
	applyRealtimeLayout()
	if realtimeActive then
		passLabel.Text = "REALTIME " .. RT_FULL_W .. "x" .. RT_FULL_H
	end
end

applyRealtimeResolution(settings.rtResIndex or 1)

for _, tile in ipairs(tiles) do
	tile.doneEvent.Event:Connect(function(data)
		-- Realtime frame tile coming back — handled independently of the progressive
		-- render state (isRendering / passNumber guards don't apply).
		if data.kind == "realtime" then
			local rt = rtImages[data.tileIndex]
			rt.image:WritePixelsBuffer(Vector2.new(data.chunkX, data.chunkY), Vector2.new(data.chunkW, data.chunkH), data.colorBuffer)
			realtimePending -= 1
			if realtimePending <= 0 then
				realtimeBusy = false
				local now = os.clock()
				if rtLastFrame > 0 then
					local dt = now - rtLastFrame
					if dt > 0 then
						local inst = 1 / dt
						rtFps = (rtFps == 0) and inst or (rtFps * 0.85 + inst * 0.15)  -- EMA smoothing
					end
				end
				rtLastFrame = now
				passLabel.Text = string.format("REALTIME %dx%d  %.0f fps", RT_FULL_W, RT_FULL_H, rtFps)
			end
			return
		end

		if not isRendering then return end
		if data.passNumber ~= passNumber then return end

		-- Parallel reconstruction result coming back from a worker Actor.
		if data.kind == "reconstruct" then
			local t = tiles[data.tileIndex]
			t.reconvBuffer = data.reconvBuffer     -- keep updated freeze state
			t.frozenCount  = data.frozenCount      -- for the G skip next time
			t.displayBuffer = data.displayBuffer   -- keep persistent display (frozen pixels)
			t.editableImage:WritePixelsBuffer(Vector2.new(0, 0), Vector2.new(TILE_W, TILE_H), data.displayBuffer)
			reconstructsPending -= 1
			if reconstructsPending <= 0 then
				finishPass()
			end
			return
		end

		local t0 = os.clock()
		local t  = tiles[data.tileIndex]

		updateAccum(t, data.colorBuffer, data.depthBuffer, data.normalBuffer, data.chunkX, data.chunkY, data.chunkW, data.chunkH, passNumber)
		chunksComplete += 1

		-- Mid-pass preview: simple bilinear blit of the chunk (no reconstruction)
		if not UPSCALE_ENABLED and (chunksComplete % DISPLAY_EVERY == 0 or chunksComplete >= totalChunks) then
			flushChunkToScreen(t, data.chunkX, data.chunkY, data.chunkW, data.chunkH)
		end

		if chunksComplete >= totalChunks then
			if UPSCALE_ENABLED and passNumber % 4 == 0 then
				-- Reconstruction runs on the workers now; finishPass fires when they return.
				dispatchParallelReconstruct()
			else
				if not UPSCALE_ENABLED then
					for _, dt in ipairs(tiles) do
						if passNumber % 4 == 0 then
							computeBloom(dt)
							flushTileWithBloom(dt)
						else
							flushTileNoBloom(dt)
						end
					end
				end
				finishPass()
			end
		else
			assignNextChunk(data.workerIndex)
		end

		local ht = os.clock() - t0
		if ht > 0.05 then
			print("SLOW HANDLER: "..string.format("%.3f",ht).."s pass="..passNumber.." chunks="..chunksComplete)
		end
	end)
end

local renderStopped = false

-- Build the full renderData (camera basis + scene lighting) for a given camera CFrame.
function buildRenderData(cf, characterFilter)
	local fovRad     = math.rad(camera.FieldOfView)
	local aspect     = FULL_W / FULL_H
	local halfHeight = math.tan(fovRad / 2)
	local halfWidth  = aspect * halfHeight
	renderData = {
		camPos          = cf.Position,
		camRight        = cf.RightVector,
		camUp           = cf.UpVector,
		camLook         = cf.LookVector,
		halfWidth       = halfWidth,
		halfHeight      = halfHeight,
		lights          = gatherLights(),
		sunStrength     = Lighting.Brightness * 0.6,
		sunDirection    = getSunDirection(),
		sunColor        = getSunColor(),
		skyBrightness   = getSkyBrightness(),
		characterFilter = characterFilter or {},
	}
end


local function startRender()
	if isRendering then return end
	isRendering   = true
	renderStopped = false
	passNumber    = 0
	startTime     = os.clock()
	resetAccumBuffers()

	camera.CameraType = Enum.CameraType.Scriptable
	task.wait(0.1)

	local characterFilter = {}
	if not settings.showPlayer and player.Character then
		-- Pass the character Model directly; Roblox excludes all its descendants
		table.insert(characterFilter, player.Character)
	end

	buildRenderData(camera.CFrame, characterFilter)
	startNewPass()
end

local function stopRender()
	if not renderStopped then
		isRendering    = false
		renderStopped  = true
		passLabel.Text = "Stopped at Pass " .. passNumber
		camera.CameraType = Enum.CameraType.Custom
		for _, w in ipairs(workers) do w.busy = false end
	else
		for _, tile in ipairs(tiles) do
			local blank = buffer.create(TILE_W * TILE_H * 4)
			buffer.fill(blank, 0, 0, TILE_W * TILE_H * 4)
			tile.editableImage:WritePixelsBuffer(Vector2.new(0, 0), Vector2.new(TILE_W, TILE_H), blank)
		end
		renderStopped  = false
		passLabel.Text = "Pass 0"
	end
end

-- ─── Realtime loop ───────────────────────────────────────────────────────────
-- Each frame, if the previous frame's workers are done, snapshot the live camera and
-- dispatch a fresh 320x180 frame across all 8 workers (2 per tile, top/bottom halves).
-- The camera is NOT locked — move it with normal controls and the view updates.
local function dispatchRealtimeFrame()
	local cf          = camera.CFrame
	local fovRad      = math.rad(camera.FieldOfView)
	local aspect      = RT_FULL_W / RT_FULL_H
	local halfHeight  = math.tan(fovRad / 2)
	local halfWidth   = aspect * halfHeight
	-- Realtime is a navigation view, so the player's character is shown (not filtered).
	local charFilter  = {}
	local halfH = RT_TILE_H // 2

	-- Camera-move detection → hard reset of the temporal accumulator.
	local moved = true
	if rtLastCamCF then
		if (cf.Position - rtLastCamCF.Position).Magnitude < 1e-3
			and cf.LookVector:Dot(rtLastCamCF.LookVector) > 0.999999 then
			moved = false
		end
	end
	rtLastCamCF = cf

	if moved then
		rtFrameN = 1
	else
		rtFrameN = math.min(rtFrameN + 1, RT_MAX_ACCUM)
	end
	rtFrameCounter += 1

	realtimePending = 0
	for ti = 1, NUM_TILES do
		local rto = rtTileOffsets[ti]
		for h = 0, 1 do   -- top/bottom half → the tile's two workers
			realtimePending += 1
			local wi = (ti - 1) * 2 + h + 1
			workers[wi].actor:SendMessage("RealtimeChunk", {
				kind            = "realtime",
				tileIndex       = ti,
				chunkX          = 0,
				chunkY          = h * halfH,
				chunkW          = RT_TILE_W,
				chunkH          = halfH,
				tileOffsetX     = rto.x,
				tileOffsetY     = rto.y,
				FULL_W          = RT_FULL_W,
				FULL_H          = RT_FULL_H,
				camPos          = cf.Position,
				camRight        = cf.RightVector,
				camUp           = cf.UpVector,
				camLook         = cf.LookVector,
				halfWidth       = halfWidth,
				halfHeight      = halfHeight,
				lights          = rtLights,
				sunStrength     = Lighting.Brightness * 0.6,
				sunDirection    = getSunDirection(),
				sunColor        = getSunColor(),
				skyBrightness   = getSkyBrightness(),
				exposure        = settings.exposure,
				bounceStrength  = settings.bounceStrength,
				sunSampleProb   = settings.sunSampleProb,
				maxBounces      = settings.rtBounces,
				rayTrace        = settings.rtRayTrace,
				characterFilter = charFilter,
				workerIndex     = wi,
				-- temporal accumulation
				frameN          = rtFrameN,
				reset           = moved,
				frame           = rtFrameCounter,
				spp             = settings.rtSamples,
			})
		end
	end
	realtimeBusy = true
end

RunService.RenderStepped:Connect(function(dt)
	-- Pick up a resolution-step change from the slider (only when idle, to avoid
	-- rebuilding images mid-frame).
	if settings.rtResIndex ~= rtAppliedIndex and not realtimeBusy then
		applyRealtimeResolution(settings.rtResIndex)
	end
	-- Pick up a pixelated/bilinear toggle change.
	if settings.rtPixelated ~= rtAppliedPixel then
		rtAppliedPixel = settings.rtPixelated
		local mode = settings.rtPixelated and Enum.ResamplerMode.Pixelated or Enum.ResamplerMode.Default
		for i = 1, NUM_TILES do
			if rtImages[i] and rtImages[i].label then rtImages[i].label.ResampleMode = mode end
		end
	end
	-- Pick up a window-mode change (fullscreen / windowed / 1:1).
	if settings.rtWindowMode ~= rtAppliedWindow then
		applyRealtimeLayout()
	end
	if realtimeActive and not realtimeBusy then
		dispatchRealtimeFrame()
	end
end)

local function toggleRealtime()
	realtimeActive = not realtimeActive
	for i = 1, NUM_TILES do rtImages[i].label.Visible = realtimeActive end
	if realtimeActive then
		isRendering   = false     -- stop any progressive render
		renderStopped = false
		realtimeBusy  = false
		rtLights      = gatherLights()   -- cache once; not re-scanned per frame
		rtLastFrame   = 0
		rtFps         = 0
		rtFrameN      = 0
		rtLastCamCF   = nil               -- forces reset on first frame (worker re-inits its accum)
		camera.CameraType = Enum.CameraType.Custom
		applyRealtimeLayout()            -- position images + show backdrop if windowed/1:1
		passLabel.Text = "REALTIME " .. RT_FULL_W .. "x" .. RT_FULL_H
	else
		rtBackdrop.Visible = false
		passLabel.Text = "Realtime off"
	end
end

-- ─── Input ─────────────────────────────────────────────────────────────────

UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end

	if input.KeyCode == Enum.KeyCode.T then
		toggleRealtime()
	end

	if input.KeyCode == Enum.KeyCode.H then
		uiHidden = not uiHidden
		settingsPanel.Visible = uiHidden and false or settingsPanel.Visible
		settingsBtn.Visible   = not uiHidden
	end

	if input.KeyCode == Enum.KeyCode.R then
		if isRendering then
			-- R while rendering: stop and pause
			stopRender()
		elseif renderStopped then
			-- R while paused: clear tiles
			for _, tile in ipairs(tiles) do
				local blank = buffer.create(TILE_W * TILE_H * 4)
				buffer.fill(blank, 0, 0, TILE_W * TILE_H * 4)
				tile.editableImage:WritePixelsBuffer(Vector2.new(0, 0), Vector2.new(TILE_W, TILE_H), blank)
			end
			renderStopped = false
			passLabel.Text = "Press R to render"
		else
			-- R while idle: start
			startRender()
		end
	end
end)

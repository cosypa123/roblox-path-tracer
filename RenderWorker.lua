--!native
--!optimize 2
-- Native code generation + aggressive optimization: compiles the numeric hot
-- path (the ray-trace / path-trace hot loop) to machine code instead of interpreted bytecode.

local actor = script:GetActor()

local primaryParams = RaycastParams.new()
local shadowParams  = RaycastParams.new()

local TRACE_DISTANCE      = 2000
local MAX_SHADOW_DISTANCE = 500
local SUN_SIZE_THRESHOLD  = 0.994
local SPECULAR_STRENGTH   = 0.12
local FRESNEL_STRENGTH    = 0.35  -- dielectric grazing-angle sky-sheen amount (bump for more)

local BOUNCE_STRENGTH     = 0.6
local SKY_ZENITH          = Vector3.new(0.10, 0.38, 0.82)
local SKY_MID             = Vector3.new(0.28, 0.60, 0.90)
local SKY_HORIZON         = Vector3.new(0.62, 0.82, 0.95)
local GROUND_BOUNCE       = Vector3.new(0.32, 0.24, 0.12)
local SUN_DIRECTION       = Vector3.new(-0.3, 0.85, 0.4).Unit
local SUN_COLOR           = Vector3.new(1.0, 0.85, 0.55)
local SKY_BRIGHTNESS      = 1.0
-- HDR encoding ceiling. MUST match the controller's HDR_MAX and the per-sample
-- clamp (maxVal) below. Encodes value/HDR_MAX*255; controller decodes inverse.
local HDR_MAX             = 8.0
local MOON_COLOR          = Vector3.new(0.55, 0.65, 0.90)
local MOON_STRENGTH       = 0.18
local SUN_SAMPLE_PROB     = 0.25
local AO_RAYS             = 3
local MAX_BOUNCES         = 3
local RT_AMBIENT_MUL      = 1.0   -- ambient scale; raised in realtime ray-trace mode (no GI bounce)

-- Per-chunk adaptive AO ray count. Set by the RenderChunk handler before it
-- calls tracePath, read by tracePath's computeAO call. Module-level so the
-- top-level tracePath function can see it (a local inside the handler closure
-- would be invisible to tracePath, which is why adaptive AO was previously
-- dead code and every hit silently used the full AO_RAYS budget).
local ADAPTIVE_AO_RAYS    = AO_RAYS

local latestPass = 0
local doneEvent  = nil

local function clamp01(x)
	return math.clamp(x, 0, 1)
end

local function safeUnit(v)
	local m = v.Magnitude
	if m < 1e-6 then return Vector3.new(0, 1, 0) end
	return v / m
end

local function makeBasis(n)
	local up        = math.abs(n.Y) < 0.999 and Vector3.new(0, 1, 0) or Vector3.new(1, 0, 0)
	local tangent   = safeUnit(up:Cross(n))
	local bitangent = safeUnit(n:Cross(tangent))
	return tangent, bitangent
end

local function schlick(cosTheta, f0)
	local m  = clamp01(1 - cosTheta)
	local m2 = m * m
	return f0 + (1 - f0) * (m2 * m2 * m)
end

-- Index of refraction for Glass (air ≈ 1.0).
local GLASS_IOR = 1.5

-- Snell's-law refraction. I = incident unit dir, N = surface normal (outward).
-- Handles both entering (ray meets front face) and exiting (ray inside the
-- medium) by flipping the normal, and returns nil on total internal reflection
-- so the caller can fall back to a straight/ reflected path.
local function refract(I, N, ior)
	local cosi = I:Dot(N)
	local n1, n2, nn
	if cosi < 0 then
		-- Entering: normal already faces the incoming ray
		cosi = -cosi
		n1, n2, nn = 1.0, ior, N
	else
		-- Exiting: flip the normal to face the ray
		n1, n2, nn = ior, 1.0, -N
	end
	local eta = n1 / n2
	local k   = 1 - eta * eta * (1 - cosi * cosi)
	if k < 0 then return nil end  -- total internal reflection
	return safeUnit(I * eta + nn * (eta * cosi - math.sqrt(k)))
end

-- returns albedoMul, ambientBoost, specPower, roughness, f0
local function getMaterialProfile(part)
	local mat = part.Material
	if mat == Enum.Material.Grass then
		return Vector3.new(0.96, 1.00, 0.96), 0.00, 8,  0.12, 0.02
	elseif mat == Enum.Material.LeafyGrass then
		return Vector3.new(0.94, 1.00, 0.93), 0.00, 8,  0.12, 0.02
	elseif mat == Enum.Material.Wood then
		return Vector3.new(1.00, 0.96, 0.89), 0.03, 12, 0.07, 0.04
	elseif mat == Enum.Material.Concrete then
		return Vector3.new(0.94, 0.94, 0.96), 0.04, 16, 0.04, 0.035
	elseif mat == Enum.Material.Slate then
		return Vector3.new(0.92, 0.93, 0.96), 0.05, 18, 0.03, 0.04
	elseif mat == Enum.Material.Metal then
		return Vector3.new(0.86, 0.88, 0.91), 0.16, 42, 0.02, 0.7
	elseif mat == Enum.Material.Glass then
		return Vector3.new(0.95, 0.97, 1.00), 0.02, 60, 0.02, 0.08
	elseif mat == Enum.Material.SmoothPlastic then
		return Vector3.new(1.00, 1.00, 1.00), 0.08, 22, 0.03, 0.05
	elseif mat == Enum.Material.Neon then
		return nil, nil, nil, nil, nil
	else
		return Vector3.new(1.00, 1.00, 1.00), 0.02, 12, 0.04, 0.03
	end
end

local function neonBoost(part)
	local pl = part:FindFirstChildOfClass("PointLight")
	return pl and math.clamp(pl.Brightness * 1.5, 2.0, 10.0) or 4.0
end

local function skyColor(rayDir)
	local elevation = rayDir.Y
	local sunDot    = rayDir:Dot(SUN_DIRECTION)
	local isDay     = SKY_BRIGHTNESS > 0.5
	-- Sunset factor: how close to horizon is the sun (0=overhead, 1=at horizon)
	local sunsetFactor = clamp01(1.0 - SUN_DIRECTION.Y * 2.5)

	if isDay then
		-- Three-zone vertical gradient: zenith → mid → horizon
		-- Using steeper curve for stronger contrast than old power-0.5
		local tHigh = clamp01(elevation * 2.2) ^ 1.4       -- zenith zone
		local tLow  = clamp01(1.0 - elevation * 3.5) ^ 0.8 -- horizon zone

		-- Base sky: blend zenith→mid→horizon
		local sky = SKY_MID:Lerp(SKY_ZENITH, tHigh):Lerp(SKY_HORIZON, tLow)

		-- Warm horizon tint — subtle during midday, stronger near sunset
		local horizonWarm = clamp01(1.0 - elevation * 4.0) ^ 2
		local warmColor = SKY_HORIZON:Lerp(
			Vector3.new(0.95, 0.75, 0.50),  -- golden horizon at sunset
			sunsetFactor
		)
		sky = sky:Lerp(warmColor, horizonWarm * 0.45)

		-- Sun atmospheric scattering: broad warm glow around sun direction
		local sunAngle  = math.max(0, sunDot)
		local atmGlow   = sunAngle ^ 4 * 0.50
		local atmColor  = Vector3.new(1.0, 0.90, 0.75):Lerp(
			Vector3.new(1.0, 0.70, 0.40),
			sunsetFactor
		)
		sky = sky + atmColor * atmGlow

		-- Sun disc
		if sunDot > SUN_SIZE_THRESHOLD then
			local sunGlow = (sunDot - SUN_SIZE_THRESHOLD) / (1 - SUN_SIZE_THRESHOLD)
			-- Overexposed white core
			local disc = sunGlow ^ 3 * 8.0
			-- Corona: pale yellow → warm white, stronger near sunset
			local coronaColor = Vector3.new(1.0, 0.95, 0.75):Lerp(
				Vector3.new(1.0, 0.70, 0.35),
				sunsetFactor
			)
			local corona = sunGlow ^ 0.55 * 1.4
			sky = sky + Vector3.new(1, 1, 1) * disc + coronaColor * corona
		end

		-- Below horizon: fade to hazy warm ground
		if elevation < 0 then
			local groundHaze = Vector3.new(0.68, 0.72, 0.72):Lerp(
				Vector3.new(0.75, 0.65, 0.50),
				sunsetFactor
			)
			sky = sky:Lerp(groundHaze, clamp01(-elevation * 5))
		end

		return sky * SKY_BRIGHTNESS
	else
		local nightSky  = Vector3.new(0.02, 0.03, 0.08)
		local starNoise = math.abs(math.sin(rayDir.X*127.3 + rayDir.Y*311.7 + rayDir.Z*74.1) * 43758.5)
		starNoise = starNoise - math.floor(starNoise)
		if starNoise > 0.997 and elevation > 0.1 then
			nightSky += Vector3.new(0.9, 0.9, 1.0) * ((starNoise - 0.997) / 0.003 * 0.8)
		end
		local moonDir        = -SUN_DIRECTION
		local moonDot        = rayDir:Dot(moonDir)
		local moonVisibility = clamp01(-SUN_DIRECTION.Y * 3)
		if moonDot > 0.9997 and moonVisibility > 0 then
			local moonAlpha = clamp01((moonDot - 0.9997) / 0.0003)
			nightSky += Vector3.new(0.92, 0.95, 1.0) * (moonAlpha ^ 0.5 * 2.5 * moonVisibility)
		end
		if moonVisibility > 0 then
			nightSky += Vector3.new(0.50, 0.58, 0.85) * (math.exp(-math.max(0, 1-moonDot)*80) * 0.4 * moonVisibility)
		end
		if elevation < 0 then
			nightSky = nightSky:Lerp(Vector3.new(0.04, 0.03, 0.03), clamp01(-elevation * 4))
		end
		return nightSky
	end
end

local function randomHemisphereDir(normal, seed1, seed2)
	local tangent, bitangent = makeBasis(normal)
	local sinTheta = math.sqrt(seed1)
	local cosTheta = math.sqrt(1 - seed1)
	local phi      = 2 * math.pi * seed2
	return safeUnit(
		tangent   * (sinTheta * math.cos(phi)) +
			bitangent * (sinTheta * math.sin(phi)) +
			normal    * cosTheta
	)
end

local function randomConeDir(axis, halfAngle, seed1, seed2)
	local tangent, bitangent = makeBasis(axis)
	local cosA  = math.cos(halfAngle)
	local r1    = cosA + seed1 * (1 - cosA)
	local sinR1 = math.sqrt(1 - r1 * r1)
	local phi   = 2 * math.pi * seed2
	return safeUnit(
		axis      * r1 +
			tangent   * (sinR1 * math.cos(phi)) +
			bitangent * (sinR1 * math.sin(phi))
	)
end

-- Shadow-ray light transmission: 1 = fully lit, 0 = fully blocked by an opaque
-- occluder. Glass and mostly-transparent parts pass most light through, so they
-- cast a faint shadow instead of a hard black one. (Single nearest hit only — a
-- glass pane in front of an opaque wall reports the glass; acceptable here.)
local function shadowTransmit(result, sourcePart)
	if not result then return 1 end
	local inst = result.Instance
	if inst == sourcePart then return 1 end
	if inst.Transparency >= 1 then return 1 end
	if inst.Material == Enum.Material.Glass then return 0.35 end
	if inst.Transparency > 0.5 then return inst.Transparency end
	return 0
end

local function computeAO(hitPos, normal, seed1, seed2, adaptiveRays)
	local rays = adaptiveRays or AO_RAYS
	if rays == 0 then return 1.0 end
	local tangent, bitangent = makeBasis(normal)
	local aoRadius = 8
	local occluded = 0
	for i = 1, rays do
		local s1   = (seed1 * i * 127.1) % 1
		local s2   = (seed2 * i * 311.7) % 1
		local sinT = math.sqrt(s1)
		local phi  = 2 * math.pi * s2
		local aoDir = safeUnit(
			tangent   * (sinT * math.cos(phi)) +
				bitangent * (sinT * math.sin(phi)) +
				normal    * math.sqrt(1 - s1)
		)
		if workspace:Raycast(hitPos + normal * 0.05, aoDir * aoRadius, shadowParams) then
			occluded += 1
		end
	end
	return 1 - (occluded / rays) * 0.25
end

local function importanceSampleDir(normal, s1, s2, s3)
	if s3 < SUN_SAMPLE_PROB and SUN_DIRECTION:Dot(normal) > 0 then
		local dir = randomConeDir(SUN_DIRECTION, 0.15, s1, s2)
		if dir:Dot(normal) > 0 then
			return dir, math.min(1 / SUN_SAMPLE_PROB, 2.0)
		end
	end
	return randomHemisphereDir(normal, s1, s2), 1.0
end

-- NEE bounce sampler: pure cosine hemisphere, no sun bias.
-- Used for bounce rays since directLight already accounts for sun via explicit shadow ray.
-- Avoids double-counting sun contribution at each bounce depth.
local function bounceSampleDir(normal, s1, s2)
	return randomHemisphereDir(normal, s1, s2), 1.0
end

local function directLight(hitPos, normal, albedo, ambientBoost, specPower, roughness, viewDir, sunStrength, seed1, seed2, lights)
	local isDay          = SKY_BRIGHTNESS > 0.5
	local moonVisibility = clamp01(-SUN_DIRECTION.Y * 3)
	local upFactor       = clamp01(normal.Y * 0.5 + 0.5)
	local skyDir         = Vector3.new(normal.X*0.3, math.max(0.1, normal.Y), normal.Z*0.3).Unit

	local skyAmbient
	if isDay then
		skyAmbient = skyColor(skyDir) * upFactor * 0.45 * SKY_BRIGHTNESS
	else
		skyAmbient = (Vector3.new(0.10, 0.13, 0.22)*moonVisibility + Vector3.new(0.04, 0.04, 0.06)) * upFactor
	end

	local groundAmbient = GROUND_BOUNCE * (1 - upFactor) * (isDay and 0.7 or 0.05)
	-- RT_AMBIENT_MUL > 1 in realtime ray-trace mode: there's no GI bounce to fill shadows,
	-- so the deterministic sky/ground ambient is boosted to keep shadowed areas looking good.
	local ambient = Vector3.new(
		albedo.X * (skyAmbient.X + groundAmbient.X + ambientBoost) * RT_AMBIENT_MUL,
		albedo.Y * (skyAmbient.Y + groundAmbient.Y + ambientBoost) * RT_AMBIENT_MUL,
		albedo.Z * (skyAmbient.Z + groundAmbient.Z + ambientBoost) * RT_AMBIENT_MUL
	)

	local sunDiffuse   = math.max(0, normal:Dot(SUN_DIRECTION))
	local shadowFactor = 0
	if sunDiffuse > 0 and isDay then
		local sunRay = randomConeDir(SUN_DIRECTION, 0.02, seed1, seed2)
		local sunHit = workspace:Raycast(hitPos + normal * 0.03, sunRay * MAX_SHADOW_DISTANCE, shadowParams)
		shadowFactor = shadowTransmit(sunHit, nil)
	end

	local direct = Vector3.new(
		albedo.X * SUN_COLOR.X,
		albedo.Y * SUN_COLOR.Y,
		albedo.Z * SUN_COLOR.Z
	) * (sunDiffuse * sunStrength * shadowFactor)

	local reflectDir = safeUnit((2 * normal * normal:Dot(SUN_DIRECTION)) - SUN_DIRECTION)
	local spec       = math.pow(math.max(0, viewDir:Dot(reflectDir)), math.max(1, specPower * (1 - roughness)))
	local specular   = Vector3.new(1, 1, 1) * (spec * SPECULAR_STRENGTH * shadowFactor)

	local moonLight = Vector3.zero
	if moonVisibility > 0 and not isDay then
		local moonDir     = -SUN_DIRECTION
		local moonDiffuse = math.max(0, normal:Dot(moonDir))
		if moonDiffuse > 0 then
			local moonRay = randomConeDir(moonDir, 0.04, seed2, seed1)
			local moonHit = workspace:Raycast(hitPos + normal * 0.03, moonRay * MAX_SHADOW_DISTANCE, shadowParams)
			local moonT = shadowTransmit(moonHit, nil)
			if moonT > 0 then
				moonLight = Vector3.new(
					albedo.X * MOON_COLOR.X,
					albedo.Y * MOON_COLOR.Y,
					albedo.Z * MOON_COLOR.Z
				) * (moonDiffuse * MOON_STRENGTH * moonVisibility * moonT)
			end
		end
	end

	local pointLight = Vector3.zero
	for _, light in ipairs(lights) do
		if light.type ~= "directional" then
			local toLight = light.position - hitPos
			local dist    = toLight.Magnitude
			if dist <= light.range then
				local lightDir   = safeUnit(toLight)
				local NdotL      = math.max(0.05, normal:Dot(lightDir))
				local shadowDist = math.max(0, dist - 1.0)
				local shadowResult = workspace:Raycast(hitPos + normal * 0.03, lightDir * shadowDist, shadowParams)
				local lightT = (shadowDist < 0.5) and 1 or shadowTransmit(shadowResult, light.sourcePart)
				if lightT > 0 then
					local atten    = math.max(0, 1 - (dist / light.range)) ^ 2
					local strength = light.brightness * atten * 5.0 * lightT
					if light.type == "spot" then
						local spotDot  = lightDir:Dot(-light.direction)
						local cosAngle = math.cos(light.angle)
						if spotDot >= cosAngle then
							strength *= math.clamp((spotDot - cosAngle) / 0.1, 0, 1)
						else
							strength = 0
						end
					end
					pointLight += Vector3.new(
						albedo.X * light.color.X,
						albedo.Y * light.color.Y,
						albedo.Z * light.color.Z
					) * (NdotL * strength)
				end
			end
		end
	end

	return ambient + direct + specular + moonLight + pointLight
end

local tracePath

local function shadeHit(hitPart, hitPos, normal, rayDir, sunStrength, lights, s1, s2, s3, s4, depth, throughput)
	throughput = throughput or 1.0
	if hitPart.Transparency >= 1 then
		return skyColor(rayDir)
	end
	if hitPart.Material == Enum.Material.Neon then
		local c     = hitPart.Color
		local boost = neonBoost(hitPart)
		return Vector3.new(c.R * boost, c.G * boost, c.B * boost)
	end
	local albedoMul, ambientBoost, specPower, roughness, f0 = getMaterialProfile(hitPart)
	local base    = hitPart.Color
	local albedo  = Vector3.new(base.R * albedoMul.X, base.G * albedoMul.Y, base.B * albedoMul.Z)
	local viewDir = safeUnit(-rayDir)
	local direct  = directLight(hitPos, normal, albedo, ambientBoost, specPower, roughness, viewDir, sunStrength, s1, s2, lights)

	if depth > 0 then
		local albedoLum = albedo.X * 0.2126 + albedo.Y * 0.7152 + albedo.Z * 0.0722
		local pathThroughput = throughput * albedoLum
		local rrProb    = math.max(0.1, math.min(pathThroughput, 0.95))
		local bounceColor = Vector3.zero
		if s4 < rrProb then
			-- NEE: pure cosine bounce, sun already counted by directLight
			local bounceDir, isWeight = bounceSampleDir(normal, s3, s4)
			bounceColor = tracePath(hitPos + normal * 0.03, bounceDir, sunStrength, lights, s2, s3, s4, s1, depth - 1, nil, pathThroughput) * (isWeight / rrProb)
		end
		return direct + Vector3.new(
			bounceColor.X * albedo.X,
			bounceColor.Y * albedo.Y,
			bounceColor.Z * albedo.Z
		) * BOUNCE_STRENGTH
	end

	return direct
end

tracePath = function(rayOrigin, rayDir, sunStrength, lights, s1, s2, s3, s4, depth, ignoreInst, throughput)
	depth = depth or MAX_BOUNCES
	throughput = throughput or 1.0
	local result = workspace:Raycast(rayOrigin, rayDir * TRACE_DISTANCE, primaryParams)

	-- Skip self-hits on a specific instance (used by glass refraction to ignore the
	-- glass part's own back face). Advances the ray past the hit and re-casts, rather
	-- than modifying RaycastParams — which is not safe to touch in the parallel phase.
	if ignoreInst then
		local guard = 0
		while result and result.Instance == ignoreInst and guard < 4 do
			result = workspace:Raycast(result.Position + rayDir * 0.05, rayDir * TRACE_DISTANCE, primaryParams)
			guard += 1
		end
	end

	if not result then
		return skyColor(rayDir), 9999, Vector3.new(0, 1, 0)
	end

	local hitPart = result.Instance
	local hitPos  = result.Position
	local normal  = result.Normal

	local transparency = hitPart.Transparency
	if transparency >= 1 then
		return tracePath(hitPos + rayDir * 0.05, rayDir, sunStrength, lights, s1, s2, s3, s4, depth, nil, throughput)
	end
	local hitDist = (hitPos - rayOrigin).Magnitude

	if hitPart.Material == Enum.Material.Neon then
		local c         = hitPart.Color
		local boost     = neonBoost(hitPart)
		local neonColor = Vector3.new(c.R * boost, c.G * boost, c.B * boost)
		if transparency > 0 then
			local behind = tracePath(hitPos + rayDir * 0.05, rayDir, sunStrength, lights, s1, s2, s3, s4, depth, nil, throughput)
			return neonColor:Lerp(behind, transparency), hitDist, normal
		end
		return neonColor, hitDist, normal
	end

	local albedoMul, ambientBoost, specPower, roughness, f0 = getMaterialProfile(hitPart)
	local base   = hitPart.Color
	local albedo = Vector3.new(base.R * albedoMul.X, base.G * albedoMul.Y, base.B * albedoMul.Z)

	local viewDir = safeUnit(-rayDir)
	local direct  = directLight(hitPos, normal, albedo, ambientBoost, specPower, roughness, viewDir, sunStrength, s1, s2, lights)
	-- AO only on the primary hit — on bounce hits it costs AO_RAYS raycasts each for
	-- almost no visible change (occlusion of already-diffuse indirect light).
	local ao = (depth >= MAX_BOUNCES) and computeAO(hitPos, normal, s3, s4, ADAPTIVE_AO_RAYS) or 1.0

	-- Material-driven look for Metal / Glass (see reflection & transparency below).
	local mat     = hitPart.Material
	local isMetal = mat == Enum.Material.Metal
	local isGlass = mat == Enum.Material.Glass
	-- Glass reads as see-through even when the builder left Transparency at 0.
	if isGlass and transparency < 0.6 then transparency = 0.6 end

	local behindColor         = Vector3.zero
	local doTransparencyBlend = transparency > 0.01
	if doTransparencyBlend then
		-- Glass bends the transmitted ray (refraction); other transparent surfaces
		-- pass the ray straight through. We tell tracePath to ignore THIS glass part
		-- on the transmitted ray so it doesn't immediately re-hit its own back face
		-- and refract a second time (which cancels the bend back to straight-through).
		-- Done by skipping self-hits in the ray march — NOT by touching RaycastParams,
		-- which is not safe to modify in the parallel phase.
		if isGlass then
			local throughDir = refract(rayDir, normal, GLASS_IOR) or rayDir
			behindColor = tracePath(hitPos + throughDir * 0.05, throughDir, sunStrength, lights, s1, s2, s3, s4, depth, hitPart, throughput)
		else
			behindColor = tracePath(hitPos + rayDir * 0.05, rayDir, sunStrength, lights, s1, s2, s3, s4, depth, nil, throughput)
		end
	end

	-- Reflection strength & tint (material-driven):
	--   Metal  → strongly reflective, tinted by the metal's albedo (metallic look)
	--   Glass  → angle-dependent Fresnel via Schlick (clear head-on, mirror at grazing)
	--   others → explicit Reflectance property (unchanged legacy behaviour)
	local reflectStrength
	if isMetal then
		reflectStrength = math.max(hitPart.Reflectance, 0.7)
	elseif isGlass then
		reflectStrength = schlick(math.max(0, viewDir:Dot(normal)), f0)
	else
		reflectStrength = hitPart.Reflectance > 0.1 and hitPart.Reflectance or 0
	end

	local reflectColor = Vector3.zero
	if reflectStrength > 0.02 then
		local reflectDir    = safeUnit(rayDir - 2 * normal * normal:Dot(rayDir))
		local reflectResult = workspace:Raycast(hitPos + normal * 0.03, reflectDir * TRACE_DISTANCE, primaryParams)
		if reflectResult then
			local rPart = reflectResult.Instance
			if rPart.Transparency >= 1 then
				-- Sky in reflection: pre-compensate for ACES to reduce banding
				local sc = skyColor(reflectDir)
				reflectColor = sc * 2.2  -- scale into HDR range so ACES treats it smoothly
			else
				reflectColor = shadeHit(rPart, reflectResult.Position, reflectResult.Normal, reflectDir, sunStrength, lights, s2, s3, s3, s4, depth - 1, throughput)
			end
		else
			-- Sky miss: pre-compensate for ACES
			local sc = skyColor(reflectDir)
			reflectColor = sc * 2.2
		end
		-- Metals tint their reflection by albedo; dielectrics (incl. glass) reflect untinted
		if isMetal then
			reflectColor = Vector3.new(
				reflectColor.X * albedo.X,
				reflectColor.Y * albedo.Y,
				reflectColor.Z * albedo.Z
			)
		end
	end

	local bounceColor = Vector3.zero
	if depth > 0 then
		local albedoLum = albedo.X * 0.2126 + albedo.Y * 0.7152 + albedo.Z * 0.0722
		-- Throughput RR: deep/dark paths die sooner; floor keeps the first bounce honest.
		local pathThroughput = throughput * albedoLum
		local rrProb = math.max(0.1, math.min(pathThroughput, 0.95))
		if s4 < rrProb then
			local bounceDir, isWeight = bounceSampleDir(normal, s3, s4)
			bounceColor = tracePath(hitPos + normal * 0.03, bounceDir, sunStrength, lights, s2, s3, s4, s1, depth - 1, nil, pathThroughput) * (isWeight / rrProb)
		end
	end

	local final
	if isGlass then
		-- Dielectric glass: the look is dominated by what's refracted THROUGH it,
		-- with Fresnel reflection mixed in (near-clear head-on, mirror-like at the
		-- grazing rim). NO diffuse surface term (that washed it out to a grey ball);
		-- the only surface highlight is a tight specular glint from the sun.
		-- Clear glass must NOT be darkened by the part's (usually grey) colour, or it
		-- looks smoky/opaque. Tint transmission by the colour's HUE only, normalised so
		-- the brightest channel is 1: grey/white glass stays perfectly clear, while a
		-- strongly-coloured glass part still tints whatever passes through it.
		local c    = hitPart.Color
		local maxc = math.max(c.R, c.G, c.B, 0.001)
		local transmit = Vector3.new(
			behindColor.X * (c.R / maxc),
			behindColor.Y * (c.G / maxc),
			behindColor.Z * (c.B / maxc)
		)
		final = transmit:Lerp(reflectColor, reflectStrength)
		-- Sharp sun glint (a reflection of the light source, not diffuse shading)
		if SKY_BRIGHTNESS > 0.5 then
			local sunReflect = safeUnit((2 * normal * normal:Dot(SUN_DIRECTION)) - SUN_DIRECTION)
			local glint = math.pow(math.max(0, viewDir:Dot(sunReflect)), 120)
			final = final + Vector3.new(1, 1, 1) * (glint * 0.9)
		end
	else
		final = (direct + Vector3.new(
			bounceColor.X * albedo.X,
			bounceColor.Y * albedo.Y,
			bounceColor.Z * albedo.Z
			) * BOUNCE_STRENGTH) * ao

		final = final:Lerp(reflectColor, reflectStrength)

		-- Dielectric Fresnel rim (grazing-angle sheen). The old all-surface Fresnel went
		-- green everywhere because it (1) added reflection ON TOP of full diffuse (energy
		-- double-count), (2) ignored roughness so matte grass rimmed as hard as glass,
		-- (3) sharp-mirror-reflected the GI-lit, saturated grass, (4) with grazing angles
		-- covering most of the frame. This version fixes all four:
		--   • reflects the sky ANALYTICALLY — never the green GI bounce
		--   • scaled by (1 - roughness)^2 — rough surfaces barely sheen, smooth ones do
		--   • folded in with :Lerp (energy-conserving: reflection trades against diffuse)
		--   • capped low so no dielectric becomes a full mirror
		if not isMetal then
			local smooth = (1 - roughness)
			local fres = schlick(math.max(0, viewDir:Dot(normal)), f0) * (smooth * smooth) * FRESNEL_STRENGTH
			fres = math.min(fres, 0.3)
			if fres > 0.01 then
				local frDir = safeUnit(rayDir - 2 * normal * normal:Dot(rayDir))
				final = final:Lerp(skyColor(frDir), fres)
			end
		end
	end

	local distance = (hitPos - rayOrigin).Magnitude
	local haze     = clamp01((distance - 300) / 800) * 0.03
	final = final:Lerp(SKY_HORIZON, haze)

	-- Glass already folded its transmission into `final` above; only non-glass
	-- transparents blend the straight-through background here (avoids double-count).
	if doTransparencyBlend and not isGlass then
		final = final:Lerp(behindColor, transparency)
	end

	return final, hitDist, normal
end

-- ─── Parallel ASR reconstruction ───────────────────────────────────────────
-- The ASR reconstruction kernel, moved off the Controller's main thread. Each tile
-- is reconstructed here in the parallel phase and the finished display buffer (plus
-- updated freeze state and frozen count) is sent back. The persistent display buffer
-- is passed in so frozen pixels keep their previous value. Kernel logic MIRRORS the
-- Controller's reconstructTile — keep the two in sync.

local R_GAMMA_INV = 1 / 2.2
local function rAces(x)
	local a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
	return math.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
end
-- Gentle filmic S-curve applied after ACES: deepens shadows and lifts highlights slightly
-- for "pop", blended in by R_CONTRAST so it stays subtle (smoothstep is the S shape).
local R_CONTRAST = 0.20
local function rGrade(x)
	local s = x * x * (3 - 2 * x)          -- smoothstep S-curve
	return x + (s - x) * R_CONTRAST        -- blend toward it, mildly
end
local R_EXPOSURE = 0.5  -- set per call from Controller settings
local function rdwrite(v, noise)
	return math.clamp(math.floor(rGrade(rAces(math.max(0, v) * R_EXPOSURE)) ^ R_GAMMA_INV * 255 + noise), 0, 255)
end

-- RGB-together film grade (geometry only). Tonemaps each channel (exposure+ACES+S-curve),
-- then applies a subtle "film" look that needs all three channels: warm the highlights,
-- cool the shadows (split-tone by luminance), and lift saturation a touch. Finally gamma +
-- quantize. Kept gentle — graded photo, not a filter.
local GRADE_WARM = 0.05    -- highlight warmth (R up / B down in brights)
local GRADE_COOL = 0.04    -- shadow coolness (B up / R down in darks)
local GRADE_SAT  = 1.10    -- saturation multiplier
local function rdwrite3(r, g, b, noise)
	-- tonemap per channel
	r = rGrade(rAces(math.max(0, r) * R_EXPOSURE))
	g = rGrade(rAces(math.max(0, g) * R_EXPOSURE))
	b = rGrade(rAces(math.max(0, b) * R_EXPOSURE))
	-- luminance for split-tone + saturation
	local lum = r * 0.2126 + g * 0.7152 + b * 0.0722
	-- split-tone: highlights warm (lum high), shadows cool (lum low)
	local warm = GRADE_WARM * lum
	local cool = GRADE_COOL * (1 - lum)
	r = r + warm - cool * 0.5
	b = b - warm + cool
	-- saturation around luminance
	r = lum + (r - lum) * GRADE_SAT
	g = lum + (g - lum) * GRADE_SAT
	b = lum + (b - lum) * GRADE_SAT
	-- gamma + quantize
	local br = math.clamp(math.floor(math.clamp(r, 0, 1) ^ R_GAMMA_INV * 255 + noise), 0, 255)
	local bg = math.clamp(math.floor(math.clamp(g, 0, 1) ^ R_GAMMA_INV * 255 + noise), 0, 255)
	local bb = math.clamp(math.floor(math.clamp(b, 0, 1) ^ R_GAMMA_INV * 255 + noise), 0, 255)
	return br, bg, bb
end
local function rdwriteSky(v, noise)
	local linear = math.max(0, v) * R_EXPOSURE
	local gamma  = math.min(1, linear) ^ R_GAMMA_INV
	return math.clamp(math.floor(gamma * 255 + noise), 0, 255)
end

local R_DEPTH_REJECT   = 0.15
local R_NORMAL_REJECT  = 0.5
local R_VAR_SHARP      = 0.02
local R_FROZEN         = 3
local R_DITHER         = nil  -- lazily built once per worker (depends only on output dims)
local RT_ACCUM         = nil  -- persistent realtime running-mean buffer (owned here, not round-tripped)

actor:BindToMessageParallel("ReconstructTile", function(data)
	R_EXPOSURE = data.exposure or 0.5
	local IW, IH = data.INT_W, data.INT_H
	local OW, OH = data.TILE_W, data.TILE_H
	local accum    = data.accumBuffer
	local depth    = data.depthBuffer
	local norm     = data.normalBuffer
	local variance = data.varianceM2
	local skyB     = data.skyBuffer
	local reconvBuf = data.reconvBuffer
	local display  = data.displayBuffer     -- persistent; frozen pixels keep their value
	local passNum  = data.passNumber

	if not R_DITHER then
		R_DITHER = buffer.create(OW * OH * 4)
		for p = 0, OW * OH - 1 do
			buffer.writef32(R_DITHER, p * 4, ((p * 2654435761) % 256) / 255 - 0.5)
		end
	end

	local filterStrength = math.max(0.05, 1.0 - passNum / 40.0)
	local invPass        = 1 / math.max(1, passNum)
	local frozenCount    = 0

	for oy = 0, OH - 1 do
		for ox = 0, OW - 1 do
			local ri = oy * OW + ox
			local rc = buffer.readu8(reconvBuf, ri)

			if rc >= R_FROZEN then
				frozenCount += 1   -- held; display already carries the frozen value
			else
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
				local noise   = buffer.readf32(R_DITHER, di)

				if cvar < R_VAR_SHARP then
					-- Converged: depth/sky-guarded bilinear of the 4 surrounding texels
					local cd  = buffer.readf32(depth, cdi0)
					local tx  = fsx - math.floor(fsx); if tx < 0 then tx = 0 elseif tx > 1 then tx = 1 end
					local ty  = fsy - math.floor(fsy); if ty < 0 then ty = 0 elseif ty > 1 then ty = 1 end
					local ix1 = math.min(ix0 + 1, IW - 1)
					local iy1 = math.min(iy0 + 1, IH - 1)
					local dTol = 0.15 * math.max(cd, 0.001)

					local sumR, sumG, sumB, sumW = 0, 0, 0, 0
					do
						local w = (1 - tx) * (1 - ty)
						sumR += cr0 * w; sumG += cg0 * w; sumB += cb0 * w; sumW += w
					end
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

					if centerSky then
						buffer.writeu8(display, di,     rdwriteSky(fr, noise))
						buffer.writeu8(display, di + 1, rdwriteSky(fg, noise))
						buffer.writeu8(display, di + 2, rdwriteSky(fb, noise))
					else
						local br, bg, bb = rdwrite3(fr, fg, fb, noise)
						buffer.writeu8(display, di,     br)
						buffer.writeu8(display, di + 1, bg)
						buffer.writeu8(display, di + 2, bb)
					end
					buffer.writeu8(display, di + 3, 255)

					rc = rc + 1
					buffer.writeu8(reconvBuf, ri, rc)
					if rc >= R_FROZEN then frozenCount += 1 end
				else
					-- Not converged: edge-aware kernel + hysteresis down-count
					local cd   = buffer.readf32(depth, cdi0)
					local cnx  = buffer.readf32(norm, ci0)
					local cny  = buffer.readf32(norm, ci0 + 4)
					local cnz  = buffer.readf32(norm, ci0 + 8)

					local radius = math.min(2, math.max(0, math.ceil(cvar * 20 * filterStrength)))
					if passNum > 30 then radius = math.min(radius, 1) end
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
							local sd = buffer.readf32(depth, sdi)
							local depthRel = math.abs(sd - cd) / math.max(cd, 0.001)
							if depthRel <= R_DEPTH_REJECT then
								local snx = buffer.readf32(norm, si)
								local sny = buffer.readf32(norm, si + 4)
								local snz = buffer.readf32(norm, si + 8)
								if cnx*snx + cny*sny + cnz*snz >= R_NORMAL_REJECT then
									local dx = ix - fsx
									local dy = iy - fsy
									local wSpatial = math.exp(-(dx*dx + dy*dy) * 0.5)
									local sr2 = buffer.readf32(accum, si)
									local sg2 = buffer.readf32(accum, si + 4)
									local sb2 = buffer.readf32(accum, si + 8)
									local colorDiff = (sr2-cr0)^2 + (sg2-cg0)^2 + (sb2-cb0)^2
									local wColor = math.exp(-colorDiff * 8.0)
									local w = wSpatial * wColor * filterStrength
										+ (1 - filterStrength) * (kx == 0 and ky == 0 and 1 or 0)
									sumR += sr2 * w; sumG += sg2 * w; sumB += sb2 * w; sumW += w
								end
							end
						end
					end

					local fr = sumW > 0 and sumR / sumW or cr0
					local fg = sumW > 0 and sumG / sumW or cg0
					local fb = sumW > 0 and sumB / sumW or cb0

					if centerSky then
						buffer.writeu8(display, di,     rdwriteSky(fr, noise))
						buffer.writeu8(display, di + 1, rdwriteSky(fg, noise))
						buffer.writeu8(display, di + 2, rdwriteSky(fb, noise))
					else
						local br, bg, bb = rdwrite3(fr, fg, fb, noise)
						buffer.writeu8(display, di,     br)
						buffer.writeu8(display, di + 1, bg)
						buffer.writeu8(display, di + 2, bb)
					end
					buffer.writeu8(display, di + 3, 255)

					if rc > 0 then buffer.writeu8(reconvBuf, ri, rc - 1) end
				end
			end
		end
	end

	task.synchronize()
	local ev = script.Parent.Parent.Parent:FindFirstChild("BindableEvent")
	ev:Fire({
		kind          = "reconstruct",
		tileIndex     = data.tileIndex,
		passNumber    = passNum,
		displayBuffer = display,
		reconvBuffer  = reconvBuf,
		frozenCount   = frozenCount,
	})
end)

-- ─── Realtime mode ──────────────────────────────────────────────────────────
-- A single low-res frame traced at 1 sample/pixel, 1 bounce, no AO, tonemapped in
-- the worker and sent back ready to blit. No accumulation, no reconstruction — the
-- camera moves freely and each frame is drawn fresh. Noisy by nature (1 spp); the
-- point is interactivity, not the converged image.
actor:BindToMessageParallel("RealtimeChunk", function(data)
	R_EXPOSURE = data.exposure or 0.5
	if data.sunDirection  then SUN_DIRECTION  = data.sunDirection  end
	if data.sunColor      then SUN_COLOR      = data.sunColor      end
	if data.skyBrightness then SKY_BRIGHTNESS = data.skyBrightness end
	BOUNCE_STRENGTH  = data.bounceStrength or 0.6
	SUN_SAMPLE_PROB  = data.sunSampleProb or 0.25
	if data.rayTrace then
		-- Ray-trace mode: no stochastic GI bounce at all → zero noise, clean every frame,
		-- and faster (one less ray/pixel). Deterministic ambient is boosted to fill shadows.
		MAX_BOUNCES     = 0
		RT_AMBIENT_MUL  = 1.6
	else
		MAX_BOUNCES     = data.maxBounces or 1
		RT_AMBIENT_MUL  = 1.0
	end
	ADAPTIVE_AO_RAYS = 0  -- no AO in realtime

	local chunkX, chunkY = data.chunkX, data.chunkY
	local chunkW, chunkH = data.chunkW, data.chunkH
	local tileOffsetX, tileOffsetY = data.tileOffsetX, data.tileOffsetY
	local FULL_W, FULL_H = data.FULL_W, data.FULL_H
	local halfWidth, halfHeight = data.halfWidth, data.halfHeight
	local lights = data.lights
	local sunStrength = data.sunStrength or 1.8
	local fixedCamFrame = CFrame.fromMatrix(data.camPos, data.camRight, data.camUp, -data.camLook)

	-- Temporal accumulation: frameN is how many frames have accumulated at the current
	-- camera pose. reset (or frameN<=1) starts fresh. The running mean (linear HDR) is kept
	-- in this worker's own RT_ACCUM between frames — not sent to/from the Controller — so we
	-- average the new sample in place, then tonemap the mean. frame varies the seed each
	-- frame so new samples differ and the average actually converges.
	local frameN    = data.frameN or 1
	local reset     = data.reset or (frameN <= 1)
	local frame     = data.frame or 0
	local spp       = data.spp or 1   -- samples per pixel per frame

	local out  = buffer.create(chunkW * chunkH * 4)   -- tonemapped display (sent back)
	-- Worker-owned running mean: kept here between frames instead of round-tripping to
	-- the Controller every frame. Reallocated (and force-reset) if the region size changed.
	local need = chunkW * chunkH * 12
	if RT_ACCUM == nil or buffer.len(RT_ACCUM) ~= need then
		RT_ACCUM = buffer.create(need)
		reset = true
	end
	local accum = RT_ACCUM

	task.synchronize()
	if data.characterFilter and #data.characterFilter > 0 then
		primaryParams.FilterType = Enum.RaycastFilterType.Exclude
		primaryParams.FilterDescendantsInstances = data.characterFilter
		shadowParams.FilterType = Enum.RaycastFilterType.Exclude
		shadowParams.FilterDescendantsInstances = data.characterFilter
	else
		primaryParams.FilterType = Enum.RaycastFilterType.Exclude
		primaryParams.FilterDescendantsInstances = {}
		shadowParams.FilterType = Enum.RaycastFilterType.Exclude
		shadowParams.FilterDescendantsInstances = {}
	end
	task.desynchronize()

	local invN = 1 / frameN
	for ly = 0, chunkH - 1 do
		local py = chunkY + ly
		for lx = 0, chunkW - 1 do
			local px = chunkX + lx
			local gx = tileOffsetX + px
			local gy = tileOffsetY + py
			-- Samples per frame: K jittered samples, averaged. This is the lever that
			-- cuts noise WHILE MOVING (temporal accumulation only helps when still).
			-- Sampling is low-discrepancy (R-sequence frac(n·α)) decorrelated per pixel by a
			-- stable hash offset (Cranley-Patterson). vs the old white-noise sin-hash this
			-- spreads error evenly instead of clumping → the same ray count looks cleaner,
			-- which is the whole point while moving (no accumulation to hide clumps).
			local pid = gy * FULL_W + gx
			local o1 = math.abs(math.sin(pid * 12.9898) * 43758.5453) % 1
			local o2 = math.abs(math.sin(pid * 78.2330) * 43758.5453) % 1
			local o3 = math.abs(math.sin(pid * 39.4250) * 43758.5453) % 1
			local o4 = math.abs(math.sin(pid * 93.9890) * 43758.5453) % 1
			local sr, sg, sb = 0, 0, 0
			local hitDepth = 0
			for si = 1, spp do
				local sidx = frame * spp + si       -- unique per frame+sample (→ convergence when still)
				local s1 = (sidx * 0.7548776662 + o1) % 1
				local s2 = (sidx * 0.5698402910 + o2) % 1
				local s3 = (sidx * 0.3819660113 + o3) % 1
				local s4 = (sidx * 0.6180339887 + o4) % 1
				-- Subpixel jitter → anti-aliasing + convergence
				local screenX = (((gx + s1 - 0.5) + 0.5) / FULL_W) * 2 - 1
				local screenY = -((((gy + s2 - 0.5) + 0.5) / FULL_H) * 2 - 1)
				local viewVector = Vector3.new(screenX * halfWidth, screenY * halfHeight, -1).Unit
				local rayDir = fixedCamFrame:VectorToWorldSpace(viewVector).Unit
				local sample, hd = tracePath(fixedCamFrame.Position, rayDir, sunStrength, lights, s1, s2, s3, s4)
				sr += math.min(sample.X, 8.0)
				sg += math.min(sample.Y, 8.0)
				sb += math.min(sample.Z, 8.0)
				hitDepth = hd
			end
			local invSpp = 1 / spp
			local r = sr * invSpp
			local g = sg * invSpp
			local b = sb * invSpp

			local pi  = (ly * chunkW + lx)
			local ao12 = pi * 12
			local ar, ag, ab
			if reset then
				ar, ag, ab = r, g, b
			else
				ar = buffer.readf32(accum, ao12)     * (frameN - 1) * invN + r * invN
				ag = buffer.readf32(accum, ao12 + 4) * (frameN - 1) * invN + g * invN
				ab = buffer.readf32(accum, ao12 + 8) * (frameN - 1) * invN + b * invN
			end
			buffer.writef32(accum, ao12,     ar)
			buffer.writef32(accum, ao12 + 4, ag)
			buffer.writef32(accum, ao12 + 8, ab)


			local off = pi * 4
			if hitDepth >= 9990 then
				buffer.writeu8(out, off,     rdwriteSky(ar, 0))
				buffer.writeu8(out, off + 1, rdwriteSky(ag, 0))
				buffer.writeu8(out, off + 2, rdwriteSky(ab, 0))
			else
				local br, bg, bb = rdwrite3(ar, ag, ab, 0)
				buffer.writeu8(out, off,     br)
				buffer.writeu8(out, off + 1, bg)
				buffer.writeu8(out, off + 2, bb)
			end
			buffer.writeu8(out, off + 3, 255)
		end
	end

	task.synchronize()
	local ev = script.Parent.Parent.Parent:FindFirstChild("BindableEvent")
	ev:Fire({
		kind          = "reconstruct",
		tileIndex     = data.tileIndex,
		passNumber    = passNum,
		displayBuffer = display,
		reconvBuffer  = reconvBuf,
		frozenCount   = frozenCount,
	})
end)

-- ─── Realtime mode ──────────────────────────────────────────────────────────
-- A single low-res frame traced at 1 sample/pixel, 1 bounce, no AO, tonemapped in
-- the worker and sent back ready to blit. No accumulation, no reconstruction — the
-- camera moves freely and each frame is drawn fresh. Noisy by nature (1 spp); the
-- point is interactivity, not the converged image.
actor:BindToMessageParallel("RealtimeChunk", function(data)
	R_EXPOSURE = data.exposure or 0.5
	if data.sunDirection  then SUN_DIRECTION  = data.sunDirection  end
	if data.sunColor      then SUN_COLOR      = data.sunColor      end
	if data.skyBrightness then SKY_BRIGHTNESS = data.skyBrightness end
	BOUNCE_STRENGTH  = data.bounceStrength or 0.6
	SUN_SAMPLE_PROB  = data.sunSampleProb or 0.25
	if data.rayTrace then
		-- Ray-trace mode: no stochastic GI bounce at all → zero noise, clean every frame,
		-- and faster (one less ray/pixel). Deterministic ambient is boosted to fill shadows.
		MAX_BOUNCES     = 0
		RT_AMBIENT_MUL  = 1.6
	else
		MAX_BOUNCES     = data.maxBounces or 1
		RT_AMBIENT_MUL  = 1.0
	end
	ADAPTIVE_AO_RAYS = 0  -- no AO in realtime

	local chunkX, chunkY = data.chunkX, data.chunkY
	local chunkW, chunkH = data.chunkW, data.chunkH
	local tileOffsetX, tileOffsetY = data.tileOffsetX, data.tileOffsetY
	local FULL_W, FULL_H = data.FULL_W, data.FULL_H
	local halfWidth, halfHeight = data.halfWidth, data.halfHeight
	local lights = data.lights
	local sunStrength = data.sunStrength or 1.8
	local fixedCamFrame = CFrame.fromMatrix(data.camPos, data.camRight, data.camUp, -data.camLook)

	-- Temporal accumulation: frameN is how many frames have accumulated at the current
	-- camera pose. reset (or frameN<=1) starts fresh. The running mean (linear HDR) is kept
	-- in this worker's own RT_ACCUM between frames — not sent to/from the Controller — so we
	-- average the new sample in place, then tonemap the mean. frame varies the seed each
	-- frame so new samples differ and the average actually converges.
	local frameN    = data.frameN or 1
	local reset     = data.reset or (frameN <= 1)
	local frame     = data.frame or 0
	local spp       = data.spp or 1   -- samples per pixel per frame

	local out  = buffer.create(chunkW * chunkH * 4)   -- tonemapped display (sent back)
	-- Worker-owned running mean: kept here between frames instead of round-tripping to
	-- the Controller every frame. Reallocated (and force-reset) if the region size changed.
	local need = chunkW * chunkH * 12
	if RT_ACCUM == nil or buffer.len(RT_ACCUM) ~= need then
		RT_ACCUM = buffer.create(need)
		reset = true
	end
	local accum = RT_ACCUM

	task.synchronize()
	if data.characterFilter and #data.characterFilter > 0 then
		primaryParams.FilterType = Enum.RaycastFilterType.Exclude
		primaryParams.FilterDescendantsInstances = data.characterFilter
		shadowParams.FilterType = Enum.RaycastFilterType.Exclude
		shadowParams.FilterDescendantsInstances = data.characterFilter
	else
		primaryParams.FilterType = Enum.RaycastFilterType.Exclude
		primaryParams.FilterDescendantsInstances = {}
		shadowParams.FilterType = Enum.RaycastFilterType.Exclude
		shadowParams.FilterDescendantsInstances = {}
	end
	task.desynchronize()

	local invN = 1 / frameN
	for ly = 0, chunkH - 1 do
		local py = chunkY + ly
		for lx = 0, chunkW - 1 do
			local px = chunkX + lx
			local gx = tileOffsetX + px
			local gy = tileOffsetY + py
			-- Samples per frame: K jittered samples, averaged. This is the lever that
			-- cuts noise WHILE MOVING (temporal accumulation only helps when still).
			-- Sampling is low-discrepancy (R-sequence frac(n·α)) decorrelated per pixel by a
			-- stable hash offset (Cranley-Patterson). vs the old white-noise sin-hash this
			-- spreads error evenly instead of clumping → the same ray count looks cleaner,
			-- which is the whole point while moving (no accumulation to hide clumps).
			local pid = gy * FULL_W + gx
			local o1 = math.abs(math.sin(pid * 12.9898) * 43758.5453) % 1
			local o2 = math.abs(math.sin(pid * 78.2330) * 43758.5453) % 1
			local o3 = math.abs(math.sin(pid * 39.4250) * 43758.5453) % 1
			local o4 = math.abs(math.sin(pid * 93.9890) * 43758.5453) % 1
			local sr, sg, sb = 0, 0, 0
			local hitDepth = 0
			for si = 1, spp do
				local sidx = frame * spp + si       -- unique per frame+sample (→ convergence when still)
				local s1 = (sidx * 0.7548776662 + o1) % 1
				local s2 = (sidx * 0.5698402910 + o2) % 1
				local s3 = (sidx * 0.3819660113 + o3) % 1
				local s4 = (sidx * 0.6180339887 + o4) % 1
				-- Subpixel jitter → anti-aliasing + convergence
				local screenX = (((gx + s1 - 0.5) + 0.5) / FULL_W) * 2 - 1
				local screenY = -((((gy + s2 - 0.5) + 0.5) / FULL_H) * 2 - 1)
				local viewVector = Vector3.new(screenX * halfWidth, screenY * halfHeight, -1).Unit
				local rayDir = fixedCamFrame:VectorToWorldSpace(viewVector).Unit
				local sample, hd = tracePath(fixedCamFrame.Position, rayDir, sunStrength, lights, s1, s2, s3, s4)
				sr += math.min(sample.X, 8.0)
				sg += math.min(sample.Y, 8.0)
				sb += math.min(sample.Z, 8.0)
				hitDepth = hd
			end
			local invSpp = 1 / spp
			local r = sr * invSpp
			local g = sg * invSpp
			local b = sb * invSpp

			local pi  = (ly * chunkW + lx)
			local ao12 = pi * 12
			local ar, ag, ab
			if reset then
				ar, ag, ab = r, g, b
			else
				ar = buffer.readf32(accum, ao12)     * (frameN - 1) * invN + r * invN
				ag = buffer.readf32(accum, ao12 + 4) * (frameN - 1) * invN + g * invN
				ab = buffer.readf32(accum, ao12 + 8) * (frameN - 1) * invN + b * invN
			end
			buffer.writef32(accum, ao12,     ar)
			buffer.writef32(accum, ao12 + 4, ag)
			buffer.writef32(accum, ao12 + 8, ab)


			local off = pi * 4
			if hitDepth >= 9990 then
				buffer.writeu8(out, off,     rdwriteSky(ar, 0))
				buffer.writeu8(out, off + 1, rdwriteSky(ag, 0))
				buffer.writeu8(out, off + 2, rdwriteSky(ab, 0))
			else
				local br, bg, bb = rdwrite3(ar, ag, ab, 0)
				buffer.writeu8(out, off,     br)
				buffer.writeu8(out, off + 1, bg)
				buffer.writeu8(out, off + 2, bb)
			end
			buffer.writeu8(out, off + 3, 255)
		end
	end

	task.synchronize()
	local ev = script.Parent.Parent.Parent:FindFirstChild("BindableEvent")
	ev:Fire({
		kind        = "realtime",
		tileIndex   = data.tileIndex,
		chunkX      = chunkX,
		chunkY      = chunkY,
		chunkW      = chunkW,
		chunkH      = chunkH,
		colorBuffer = out,
		workerIndex = data.workerIndex,
	})
end)

actor:BindToMessageParallel("RenderChunk", function(data)
	if not doneEvent then
		doneEvent = script.Parent.Parent.Parent:FindFirstChild("BindableEvent")
	end
	if data.passNumber == 1 then
		latestPass = 0
	end
	if data.passNumber < latestPass then
		doneEvent:Fire({
			chunkX       = data.chunkX,
			chunkY       = data.chunkY,
			chunkW       = data.chunkW,
			chunkH       = data.chunkH,
			tileIndex    = data.tileIndex,
			workerIndex  = data.workerIndex,
			colorBuffer  = buffer.create(data.chunkW * data.chunkH * 4),
			depthBuffer  = buffer.create(data.chunkW * data.chunkH * 4),
			normalBuffer = buffer.create(data.chunkW * data.chunkH * 12),
			passNumber   = data.passNumber,
		})
		return
	end
	latestPass = data.passNumber

	local chunkX      = data.chunkX
	local chunkY      = data.chunkY
	local chunkW      = data.chunkW
	local chunkH      = data.chunkH
	local tileIndex   = data.tileIndex
	local tileOffsetX = data.tileOffsetX
	local tileOffsetY = data.tileOffsetY
	local workerIndex = data.workerIndex
	local FULL_W      = data.FULL_W
	local FULL_H      = data.FULL_H
	local halfWidth   = data.halfWidth
	local halfHeight  = data.halfHeight
	local lights      = data.lights
	local sunStrength = data.sunStrength or 1.8
	local passNumber  = data.passNumber or 1
	local seed        = data.seed or 0

	if data.sunDirection   then SUN_DIRECTION  = data.sunDirection   end
	if data.sunColor       then SUN_COLOR       = data.sunColor       end
	if data.skyBrightness  then SKY_BRIGHTNESS  = data.skyBrightness  end
	if data.bounceStrength then BOUNCE_STRENGTH = data.bounceStrength end
	if data.sunSampleProb  then SUN_SAMPLE_PROB = data.sunSampleProb  end
	if data.aoRays         then AO_RAYS         = data.aoRays         end
	if data.maxBounces     then MAX_BOUNCES     = data.maxBounces     end
	RT_AMBIENT_MUL = 1.0   -- progressive always uses full GI, normal ambient
	local chunkVariance = data.chunkVariance or 1.0  -- normalized variance for this chunk
	-- Adaptive AO: scale ray count from 1 (converged) to AO_RAYS (max variance)
	-- sqrt gives gentler falloff so mid-variance regions still get decent AO
	local adaptiveAORays = math.max(1, math.ceil(AO_RAYS * math.sqrt(math.min(1.0, chunkVariance * 8))))
	-- Publish to module scope so tracePath's computeAO call actually sees it.
	ADAPTIVE_AO_RAYS = adaptiveAORays
	local asrJitterX = data.asrJitterX or 0
	local asrJitterY = data.asrJitterY or 0

	local fixedCamFrame = CFrame.fromMatrix(
		data.camPos, data.camRight, data.camUp, -data.camLook
	)

	local colorBuffer  = buffer.create(chunkW * chunkH * 4)
	local depthBuffer  = buffer.create(chunkW * chunkH * 4)   -- f32 hit distance
	local normalBuffer = buffer.create(chunkW * chunkH * 12)  -- f32 xyz normal
	local bufferOffset = 0

	task.synchronize()
	if data.characterFilter and #data.characterFilter > 0 then
		primaryParams.FilterDescendantsInstances = data.characterFilter
		primaryParams.FilterType = Enum.RaycastFilterType.Exclude
		shadowParams.FilterDescendantsInstances  = data.characterFilter
		shadowParams.FilterType = Enum.RaycastFilterType.Exclude
	else
		-- No filter: Exclude with empty list = exclude nothing = hit everything
		primaryParams.FilterType = Enum.RaycastFilterType.Exclude
		primaryParams.FilterDescendantsInstances = {}
		shadowParams.FilterType = Enum.RaycastFilterType.Exclude
		shadowParams.FilterDescendantsInstances  = {}
	end
	task.desynchronize()

	for localY = 0, chunkH - 1 do
		local pixelY = chunkY + localY
		for localX = 0, chunkW - 1 do
			local pixelX = chunkX + localX

			local globalX = tileOffsetX + pixelX
			local globalY = tileOffsetY + pixelY

			-- One sample per pixel per pass; the Controller accumulates across passes.
			local accumR, accumG, accumB = 0, 0, 0
			local accumDepth, accumNX, accumNY, accumNZ = 0, 0, 0, 0
			local n = 1

			for sampleIdx = 1, n do
				local pixelSeed = ((pixelY * FULL_W + pixelX) % 9973) * ((passNumber + sampleIdx) % 997) + (seed % 1000) + sampleIdx * 7919
				local s1 = math.abs((math.sin(pixelSeed * 127.1) * 43758.5453) % 1)
				local s2 = math.abs((math.sin(pixelSeed * 311.7) * 43758.5453) % 1)
				local s3 = math.abs((math.sin(pixelSeed * 74.3)  * 43758.5453) % 1)
				local s4 = math.abs((math.sin(pixelSeed * 233.1) * 43758.5453) % 1)

				-- Apply subpixel jitter for ASR: shifts sample grid by Halton offset
				-- so accumulated passes cover the full output resolution space
				local screenX = (((globalX + s1 - 0.5 + asrJitterX) + 0.5) / FULL_W) * 2 - 1
				local screenY = -((((globalY + s2 - 0.5 + asrJitterY) + 0.5) / FULL_H) * 2 - 1)

				local viewVector = Vector3.new(screenX * halfWidth, screenY * halfHeight, -1).Unit
				local rayDir     = fixedCamFrame:VectorToWorldSpace(viewVector).Unit

				local sample, hitDepth, hitNormal = tracePath(fixedCamFrame.Position, rayDir, sunStrength, lights, s1, s2, s3, s4)

				local maxVal = 8.0
				accumR += math.min(sample.X, maxVal)
				accumG += math.min(sample.Y, maxVal)
				accumB += math.min(sample.Z, maxVal)
				accumDepth += hitDepth
				accumNX += hitNormal.X
				accumNY += hitNormal.Y
				accumNZ += hitNormal.Z
			end

			local finalR = accumR / n
			local finalG = accumG / n
			local finalB = accumB / n
			-- Alpha = 0 for sky pixels (hitDepth >= 9999), 255 for geometry
			-- Controller uses this to bypass ACES tonemapper for sky pixels
			local isSkyPixel = (accumDepth / n) >= 9990

			if isSkyPixel then
				-- Sky: encode as direct [0,1] values (no HDR scaling needed)
				buffer.writeu8(colorBuffer, bufferOffset,     math.clamp(math.round(finalR * 255), 0, 255))
				buffer.writeu8(colorBuffer, bufferOffset + 1, math.clamp(math.round(finalG * 255), 0, 255))
				buffer.writeu8(colorBuffer, bufferOffset + 2, math.clamp(math.round(finalB * 255), 0, 255))
				buffer.writeu8(colorBuffer, bufferOffset + 3, 0)
			else
				-- Geometry: HDR encoding as normal
				buffer.writeu8(colorBuffer, bufferOffset,     math.clamp(math.round(finalR / HDR_MAX * 255), 0, 255))
				buffer.writeu8(colorBuffer, bufferOffset + 1, math.clamp(math.round(finalG / HDR_MAX * 255), 0, 255))
				buffer.writeu8(colorBuffer, bufferOffset + 2, math.clamp(math.round(finalB / HDR_MAX * 255), 0, 255))
				buffer.writeu8(colorBuffer, bufferOffset + 3, 255)
			end

			-- Write depth and normal for reconstruction (from last sample)
			local pixelIdx = (localY * chunkW + localX)
			buffer.writef32(depthBuffer,  pixelIdx * 4,  accumDepth / n)
			buffer.writef32(normalBuffer, pixelIdx * 12,     accumNX / n)
			buffer.writef32(normalBuffer, pixelIdx * 12 + 4, accumNY / n)
			buffer.writef32(normalBuffer, pixelIdx * 12 + 8, accumNZ / n)

			bufferOffset += 4
		end
	end

	task.synchronize()

	doneEvent:Fire({
		chunkX       = chunkX,
		chunkY       = chunkY,
		chunkW       = chunkW,
		chunkH       = chunkH,
		tileIndex    = tileIndex,
		workerIndex  = workerIndex,
		colorBuffer  = colorBuffer,
		depthBuffer  = depthBuffer,
		normalBuffer = normalBuffer,
		passNumber   = passNumber,
	})
end)

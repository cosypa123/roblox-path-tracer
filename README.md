# Roblox CPU Path Tracer

A path tracer that runs entirely on the CPU, in Luau, inside Roblox. No GPU compute, no
shaders, no external anything. Rays traced by hand, in a scripting language, on a game
engine that was never meant to do this.

It has two modes. A progressive mode that renders a real open map at 1920x1080 with global
illumination, reflections, refractive glass, and a day/night sky, converging to a clean
image. And a realtime mode you can fly around in, tracing live as you move.

The whole thing started as one script that took 1 minute 12 seconds to render a single flat
frame on one core. The question was never "is this a good idea." It was "how far does this
go before it hits a wall." This repo is how far.

---

## What it does

- **Progressive mode.** Full path tracing: multi-bounce global illumination, next-event
  estimation, ACES tonemapping with a filmic grade, a cinematic sky. It converges to a clean
  1080p image. This is the beauty render.
- **Realtime mode.** You move the camera and it keeps tracing, live. On a real map, not a
  Cornell box. Still camera converges to full-GI quality; moving stays navigable.
- **ASR (Accumulative Super Resolution).** A custom upscaler that renders at low internal
  resolution and reconstructs to full res with an edge-aware, variance-guided kernel. Most
  of the speed comes from here.
- **Ray-tracing mode.** A toggle that drops the stochastic GI bounce for a clean, noiseless,
  deterministic image. No noise, higher framerate, and it still looks good. Path tracing
  when you want richness, ray tracing when you want speed and stability.
- **Materials.** Metal with tinted reflections, refractive Fresnel glass with soft shadows,
  emissive neon, per-material roughness.
- **Parallel.** 8 Roblox Actors render in parallel across CPU cores, with reconstruction
  also moved off the main thread.

## How it works, briefly

The screen is split into 4 tiles. Each tile is rendered by Actor workers running in
parallel Luau. Workers trace rays against the real `workspace` geometry using the engine's
raycaster, shade the hits, accumulate samples into a floating-point buffer, tonemap, and
write pixels into EditableImages that make up the display. The controller orchestrates
passes, handles the UI, and composites.

Every bit of the light transport is written in Luau. The engine is used for exactly one
thing a path tracer needs and cannot do itself in script fast enough: ray-versus-scene
intersection. Everything else, the sampling, the bounces, the materials, the tonemap, the
upscaler, is in this repo.

## Setup (Roblox Studio)

You need two scripts in two places: the worker inside an Actor in ReplicatedStorage, and
the controller inside a ScreenGui in StarterGui.

1. **Worker.**
   - In `ReplicatedStorage`, insert an `Actor`. Rename it to exactly `RenderActor`.
   - Inside `RenderActor`, insert a `LocalScript`. Paste the contents of `RenderWorker.lua`
     into it.

2. **Controller.**
   - In `StarterGui`, insert a `ScreenGui`.
   - Inside the `ScreenGui`, insert a `LocalScript`. Paste the contents of `Controller.lua`
     into it.

3. **EditableImage.** The renderer uses `AssetService:CreateEditableImage`. If that call
   errors in your place, make sure your Studio is up to date, as the display is built on
   EditableImages.

4. Press **Play**. The controller clones `RenderActor` into 8 workers on its own and builds
   the display. Nothing else to wire up.

That is the whole setup. The controller finds the worker template by name (`RenderActor`),
so the name has to match exactly.

## Controls

- **R** : start / stop / clear the progressive render.
- **T** : toggle realtime mode (fly around).
- **H** : hide / show the UI.
- **Gear icon** : open settings.

In settings you can change resolution, samples, bounces, exposure, and the realtime options
(resolution step, samples per frame, ray-tracing toggle, window mode, pixelated scaling).

## Benchmarks

Measured on an Intel i5-11400H (laptop CPU), in the Roblox client (not Studio, which runs
2 to 3 times slower for this kind of code). Map: Crossroads. The GPU is unused throughout.

**Progressive (1080p, ASR on, 8 workers, default settings: 3 bounces, 3 AO rays)**
- 80 passes in 153.6s total, averaging 1.92 s/pass.
- Pass times range from 1.46s (late, after adaptive sampling kicks in) to 3.41s (early).
- A clean, converged 1920x1080 path-traced frame in roughly two and a half minutes.

**Realtime (128x72, ray-traced, 1 sample/frame, 8 workers)**
- Average 22.6 fps over 60 seconds with the camera moving the whole time.
- Range 13 to 44 fps depending on how much scene complexity is in view.

Framerate scales with core count and clock speed. These are CPU-bound numbers; a desktop
CPU with more cores and better cooling will beat them.

## Performance notes

- Benchmark in the **Roblox client, not Studio.** Studio runs an unoptimized environment
  and is far slower for CPU-heavy code; it is not representative. On the client the same
  config can run 2 to 3 times faster.
- The realtime default is tuned for the cheapest, highest-framerate config (lowest
  resolution, 1 sample, ray-traced) so weaker devices get a usable experience out of the
  box. Turn it up if your machine has headroom.
- This is CPU-bound by design. Framerate scales with your cores and clocks, and laptops with
  weak cooling will thermal-throttle under the sustained full-core load.

## License and credit

This project is **source-available, not open source.** You can view it, download it, run it,
experiment with it, and make videos about it (including monetized videos). You must credit
**cosypa123** as the original creator, and you cannot put it in a game without prior written
permission.

Read [LICENSE.md](LICENSE.md) for the full terms. The short version: look, learn, show it
off with credit; ask first before anything commercial or before shipping it in a game.

Created by **cosypa123**. Permission requests and questions: **Discord: cosypa123**.

## Contributing

Improvements are welcome, with specific terms (ownership transfer, credit, no game
integration rights). See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

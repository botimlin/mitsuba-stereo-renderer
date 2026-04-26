# Mitsuba Stereo Renderer

**Purpose-built to generate stereo training data for deep-learning models that perceive transparent objects in indoor scenes** — glass doors, windows, partitions, drinkware on tables, and similar surfaces that defeat standard depth sensors.

It is a standalone GPU renderer that combines a Blender procedural scene randomizer with a Mitsuba 3 (`cuda_ad_rgb`) physically-based renderer. Each scene is a **closed 6-wall chamber** (floor, ceiling, four walls — fully enclosed, no open windows or skybox) populated with a table, randomized furniture, and one or more glass slabs / cups. The chamber design constrains all lighting to the in-scene back-wall LED panel plus ceiling fill lights, so the reflection / refraction physics on the glass surfaces are well-controlled and reproducible across scenes. The output is a complete, ready-to-train dataset: rectified stereo pairs paired with pixel-perfect depth, disparity, and glass-region masks, all with reproducible per-scene seeds.

Designed for training tasks such as:
- transparent-object stereo matching / depth estimation
- glass-region segmentation
- depth-completion on transparent surfaces
- ablation studies that need physically-plausible glass interactions and large scene variety

For each scene the pipeline outputs:

- **RGB stereo pair** — `left.exr` and `right.exr`
- **Depth** (left viewpoint, in meters)
- **Disparity** (in pixels, ray-depth → Z-depth corrected)
- **Glass region masks** — left view (training, aligned to GT disparity), right view, and left ∩ right intersection (strict evaluation)
- **Per-scene JSON** with camera + lighting parameters and seed

The pipeline has three stages:

1. **Blender** — random closed-room scenes with a table and 1+ glass slabs, exported to OBJ.
2. **Mitsuba 3** — render left / right / depth / disparity / mask in a single pass.
3. **EXR tooling** — convert HDR EXR outputs to viewable PNG.

---

## Directory Layout

```
mitsuba-stereo-renderer/
├── Renderer/
│   └── stereo_renderer.py               # Stereo renderer
├── modelling/
│   ├── blender_glass_randomizer.py      # Blender scene randomizer (main)
│   ├── merge_glass_objects.py           # Import a glass-cup collection from an external .blend
│   ├── check_blend_objects.py           # Diagnostic: inspect collections / verify naming contract
│   └── calibration/                     # Coordinate calibration tools
│       ├── fix_object_origins.py        # Reset origin to bottom-center, apply transforms
│       └── scale_objects_1to10.py       # 1:10 scale-down for asset packs in real-world units
├── textures/
│   ├── manifest.json                    # Texture asset index
│   ├── furniture/                       # Furniture textures
│   ├── floor/                           # Floor textures
│   └── wall/                            # Wall + ceiling shared textures
├── tools/
│   ├── exr_to_png.py                    # EXR -> PNG conversion
│   └── exr_viewer.py                    # Interactive EXR inspector
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

- Python 3.8+
- Mitsuba 3 with `cuda_ad_rgb` variant (CUDA GPU recommended)
- Drjit, NumPy, OpenCV (`cv2`)
- Blender 4.0+ (only needed for scene generation)
- Optional: `OpenEXR` / `Imath` for `tools/exr_to_png.py`

```bash
pip install -r requirements.txt
```

The renderer auto-selects from `cuda_ad_rgb` → `cuda_rgb` → `llvm_ad_rgb` → `scalar_rgb`.

---

## Coordinate System

```
Origin: ground projection of the camera center.

Axes (Scene coordinates):
  X: left(-) <-> right(+)         [horizontal]
  Y: depth   (0=camera, +=away)   [depth]
  Z: height  (0=floor,  +=up)     [height]

Scene layout:
                Y+ (depth)
                ^
                |   back wall @ Y = CHAMBER_DEPTH
                |   ============================
                |
                |   table @ Y = TABLE_DEPTH
                |     +-----+-----+
                |     | glass     |
       X- <-----+-----+-----+-----+--->  X+
                |
                +---------------------------+
                |  camera @ (0, 680, 100)   |
                +---------------------------+
```

| Parameter | Value | Description |
|---|---|---|
| `CAMERA_DEPTH` | 680 mm | Camera depth (near back wall) |
| `CAMERA_HEIGHT` | 100 mm | Camera height above floor |
| `TABLE_DEPTH` | 350 mm | Camera-to-table depth (default lookat) |
| `TABLE_SURFACE_HEIGHT` | 100 mm | Lookat height (matches camera height) |
| `CHAMBER_DEPTH` | 800 mm | Max scene depth |
| `CHAMBER_WIDTH` | 600 mm | Scene width |
| `CHAMBER_HEIGHT` | 400 mm | Ceiling height |
| `BASELINE` | 65 mm | Stereo baseline |

**OBJ → Mitsuba transform**: `Scene (X, Y, Z) → Mitsuba (X, Z, -Y)` via `scale(0.001) @ rotate([1,0,0], -90°)`.

---

## Stage 1: Blender Scene Generation

The randomizer builds a **closed chamber** (6-wall enclosure: floor, ceiling, front/back/left/right walls) sized `CHAMBER_WIDTH × CHAMBER_HEIGHT × CHAMBER_DEPTH = 600 × 400 × 800 mm`, places a table inside it, scatters furniture, and inserts 1+ glass slabs or cups. Walls and floor get random textures from the texture pool; the ceiling shares the wall texture and carries area emitters for fill lighting.

### Prerequisites

The randomizer auto-discovers candidate meshes from named **Blender collections** in your `.blend` file. Prepare these collections (each may be empty if you don't want that category):

| Collection | Purpose |
|---|---|
| `Glass`     | Glass cup objects (the renderer treats all meshes here as transparent and expects the material `Glass_Clear`) |
| `Tables`    | Candidate tables (one is picked at random per scene; or override `table.source_names` in the CONFIG dict directly) |
| `Furniture` | Floor furniture — chairs, stools, lamps, radiators, trolleys, etc. |
| `Cabinets`  | Large cabinets / shelves (rotation-locked placement) |
| `Decor`     | Tabletop decorations — vases, small props |

Any collection that is missing or empty causes that category to be skipped — the script will not crash. Use `modelling/check_blend_objects.py` to verify your `.blend` matches the contract.

The renderer requires the glass material to be exactly `Glass_Clear` (case-insensitive); name the material on every glass mesh accordingly.

### Texture Assets

Drop 4K tileable JPG / PNG textures into:

```
textures/furniture/*.{jpg,png}
textures/floor/*.{jpg,png}
textures/wall/*.{jpg,png}
```

`manifest.json` indexes them. If a texture is missing, the material falls back to flat color.

### Run

```bash
blender your_scene.blend --background --python modelling/blender_glass_randomizer.py -- \
    --seed 42 \
    --count 500 \
    --output ./scenes_output \
    --texture_dir ./textures
```

**Output**: one `scene_XXXX.obj` (with `.mtl`) per scene; textures are copied via `path_mode='COPY'`.
**Glass material is always named `Glass_Clear`** — the renderer uses exact-name matching.

### Excluding Heavy / Non-Converging Glass

If your glass asset pack contains meshes with intricate caustics (e.g. tulip-shaped wine glasses, whisky tumblers with thick bases) or extremely high vertex counts, those scenes may not converge to clean noise even at 60K+ SPP. After importing the asset pack with `merge_glass_objects.py`, drop them with `--exclude name1,name2,...`. There is no built-in blacklist — what to exclude is asset-pack-specific.

---

## Stage 2: Mitsuba Rendering

### Renderer Spec (`Renderer/stereo_renderer.py`)

| Item | Value |
|---|---|
| Mitsuba variant | `cuda_ad_rgb` (with fallbacks) |
| Integrator | `path` |
| Resolution | 640 × 480 |
| SPP | 1024 (batched 64) |
| Max ray depth | 16 |
| Sensor | Sony IMX296LQR-C (5.023 × 3.754 mm) |
| Focal length | 6 mm |
| Baseline | 65 mm |
| Horizontal FOV | 45.4° |
| Glass IOR | 1.5 (roughness 0.02) |
| Glass type | `thin` or `thick` (CLI flag) |

**Lighting**:
- Back-wall LED area emitter (camera-side wall)
- Four ceiling area emitters embedded in the ceiling mesh

**Cameras**: parallel optical axes (non-converging), with optional gentle glass-tracking lookat (`GLASS_LOOKAT_STRENGTH=0.3`, capped at ±30 mm).

### CLI: Basic Usage

```bash
# Single scene
python Renderer/stereo_renderer.py \
    --scene scene_0001.obj \
    --output ./output

# Batch render an entire directory
python Renderer/stereo_renderer.py \
    --input_dir ./scenes_output \
    --output ./output \
    --max_scenes 100

# Multi-GPU parallel
python Renderer/stereo_renderer.py \
    --input_dir ./scenes_output \
    --output ./output \
    --num_gpus 4
```

### Important Flags

| Flag | Description |
|---|---|
| `--seed N` | MC sampler seed (reproducible per-scene randomization) |
| `--spp` | Override default SPP (1024) |
| `--glass_type {thin,thick}` | Glass BSDF model |
| `--max_scenes` | Cap on number of scenes |
| `--skip N` | Skip first N scenes (multi-GPU sharding) |
| `--scene_list FILE` | Whitelist of scene names (one `scene_XXXX` per line) |
| `--no-camera-jitter` | Disable per-scene camera randomization |
| `--num_gpus N` | Multi-GPU mode (subprocess per GPU, isolated via `CUDA_VISIBLE_DEVICES`) |

**Per-scene augmentation** (seed = hash of scene name):
- `CAMERA_X_JITTER`: ±15 mm
- `CAMERA_DEPTH_JITTER`: 0~+20 mm (back-only)
- `CAMERA_HEIGHT_JITTER`: ±10 mm
- Glass-tracking lookat: clamped to ±30 mm

### Output Files (per scene)

```
{output_dir}/{scene_name}/
├── left.exr               # Left  camera RGB (float32)
├── right.exr              # Right camera RGB (float32)
├── depth.exr              # GT depth (meters, left viewpoint)
├── disparity.exr          # GT disparity (pixels)
├── glass_mask_left.png    # Glass mask, left view — pixel-aligned to depth.exr / disparity.exr (use this for training)
├── glass_mask_right.png   # Glass mask, right view (auxiliary)
├── glass_mask.png         # Pixel-wise (left ∩ right) — strict evaluation, drops monocular-only glass regions
├── params.json            # Camera poses, intensities, seed, SPP
├── {scene_name}_report.json  # Per-scene quality report
└── preview/
    ├── left.png
    └── right.png
```

#### Which mask should I use?

The depth and disparity ground truth are rendered from the **left camera viewpoint** (`depth_camera_position()` returns the left camera position). This means:

- **Training loss on glass regions** → use `glass_mask_left.png`. It is on the same pixel grid as `disparity.exr` and `depth.exr`, so a boolean indexing of disparity by this mask Just Works.
- **Strict / "two-view consistent" evaluation** → use `glass_mask.png` (intersection). It excludes glass pixels that are only visible from one camera (occluded in the other), where stereo disparity is fundamentally undefined. Note that this is a *pixel-wise* AND of the two view masks, not a disparity-warped intersection — it is conservative but cheap and works well for most cases.
- **Right-view inspection / debugging only** → `glass_mask_right.png`. Don't pair this with the left-view depth.

### Disparity Computation

Mitsuba's depth AOV gives **ray distance** (Euclidean), but disparity needs **Z distance** (perpendicular):

```
Z = ray_depth × cos(angle)
cos(angle) = focal / sqrt(focal² + dx² + dy²)
disparity = baseline × focal / Z
```

Sample disparity ranges:
- near (depth = 350 mm) → ~142 px
- far  (depth = 800 mm) → ~62 px

---

## Stage 3: EXR Preview

```bash
# Single file
python tools/exr_to_png.py output/scene_0001/left.exr

# Directory batch
python tools/exr_to_png.py --dir ./output -o ./preview

# Unified range (compare left vs. right at equivalent exposure)
python tools/exr_to_png.py --unified left.exr right.exr -o ./preview

# Interactive viewer
python tools/exr_viewer.py output/scene_0001/left.exr
```

Loader fallback chain: `OpenEXR` → `mitsuba.Bitmap` → `cv2.imread`.

---

## Glass Detection Convention

The renderer performs **exact matching** against `glass_clear` (case-insensitive), not keyword fuzzy matching, to avoid misclassifying names like `glass_table` or `glass_shelf`. Materials are partitioned into:

- **glass** → `dielectric` BSDF (IOR = 1.5, roughness = 0.02)
- **ceiling** (matches `ceiling`/`roof`/`top`/`sky`/`plafond`/`techo`) → diffuse + area emitter
- **other** → textured diffuse, fallback to flat color

Split OBJ files are written to `tempfile.mkdtemp()` and cleaned up automatically after rendering.

---

## Contact & Issues

Questions, bug reports, and feature requests are welcome:

- **GitHub Issues** — please open an issue on this repository for anything reproducible (rendering bugs, scene-generation edge cases, documentation gaps, etc.). Include your Mitsuba / Blender / GPU info and a minimal repro when possible.
- **Email the author** — for private inquiries, collaboration, or research questions, reach out via Gmail: [botimlin@gmail.com](mailto:botimlin@gmail.com).

### Need a turnkey ready-to-render bundle?

The repository ships with **code only**: it intentionally does not include a fully-populated `.blend` file or a complete texture set, because most furniture / glass / texture assets the author uses originate from third-party packs whose licenses don't allow public redistribution.

If you have an **academic or time-critical need** (paper deadline, thesis, course project, ablation reproduction) and don't have the time to source and prepare your own assets, email [botimlin@gmail.com](mailto:botimlin@gmail.com) and the author can — at his discretion — share a working bundle (`.blend` + matching textures) for **personal academic use**. The bundle is provided as-is and remains subject to the upstream licenses of each individual asset; please do not redistribute it further or include it in a public release without verifying the license of every asset yourself.

Pull requests are also welcome, especially for additional asset packs, calibration helpers, or non-glass material support.

---

## License

`Copyright (c) 2025-2026 Po-Ting Lin`
Released under the **MIT License** (see `LICENSE`).

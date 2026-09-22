# Knight Online to Minecraft Map Converter (ko2mc)

Convert Knight Online `.gtd` (terrain) and `.opd` (object) map files into playable Minecraft Java Edition worlds (Anvil format, 1.18+).

There are two converters in this package:

| CLI | Module | What it does |
|-----|--------|--------------|
| `ko2mc-zone` | `ko2mc.zone_converter` | **Main pipeline.** Resolves every OPD shape's real `.n3pmesh` parts from KO client asset folders, applies the OPD transforms and part pivots, voxelizes the meshes, grounds them on GTD terrain, and writes a JSON placement/audit report. |
| `ko2mc` | `ko2mc` (`__main__`/`converter.py`) | Legacy converter: GTD heightmap terrain plus name-pattern-matched structure approximations and event markers. Needs no client assets. |

## Requirements

- Python 3.10+
- NumPy (the only runtime dependency; NBT/Anvil files are written by `ko2mc/mc_world.py` itself)
- KO client asset folders containing `.n3pmesh` files (for `ko2mc-zone` only)

## Installation

```bash
pip install -e .            # installs the ko2mc and ko2mc-zone commands
pip install -e ".[test]"    # also installs pytest
# or, without installing the package:
pip install -r requirements.txt
```

## Zone converter (`ko2mc-zone` / `python -m ko2mc.zone_converter`)

```bash
# Full zone. Asset roots are searched in priority order (highest first).
ko2mc-zone moradon.opd \
    --asset-roots /path/to/1886-Client /path/to/ko-assets-1298 \
    -o ./worlds

# Explicit terrain file (otherwise <opd>.gtd next to the OPD is used if present)
ko2mc-zone moradon.opd --gtd moradon.gtd --asset-roots ... -o ./worlds

# Subsets for debugging: first 200 shapes / by name / within a radius
ko2mc-zone moradon.opd --limit 200 --debug-markers --asset-roots ... -o ./worlds
ko2mc-zone moradon.opd --name-filter tree --asset-roots ... -o ./worlds
ko2mc-zone moradon.opd --radius 100 --center 860 540 --asset-roots ... -o ./worlds

# Solid interiors and a detailed audit report
ko2mc-zone moradon.opd --voxel-mode solid --audit-report-level detailed \
    --asset-roots ... -o ./worlds
```

Buildings are placed first, then terrain (terrain never overwrites building blocks).

### Options

| Option | Description |
|--------|-------------|
| `opd_file` | `.opd` input file (required) |
| `-o, --output DIR` | Output directory; the world is written to `DIR/<world name>` (required) |
| `--asset-roots DIR...` | Asset root directories, highest priority first. Without them every mesh is unresolved |
| `--world-name NAME` | World name (default: OPD file stem) |
| `--voxel-size SIZE` | Voxel edge length in KO units (default `1.0`) |
| `--voxel-mode MODE` | `surface` (default): hollow shell; `hybrid`: raw SAT occupancy (thick geometry kept, no interior fill); `solid`: flood-fill enclosed interiors |
| `--fill` | Alias for `--voxel-mode solid` |
| `--block NAME` | Default block for voxels (default `minecraft:stone_bricks`; texture/shape rules usually override it) |
| `--gtd FILE` | Terrain file (default: auto-detect `<opd>.gtd`) |
| `--no-terrain` | Skip terrain even if a GTD file is found |
| `--terrain-fill-depth N` | Blocks filled below the terrain surface (default `8`) |
| `--limit N`, `--name-filter S`, `--radius R --center X Z` | Process a subset of shapes |
| `--report FILE` | Placement report path (default `<world>/placement_report.json`) |
| `--audit-report-level standard\|detailed` | `detailed` adds per-part texture names, grounding sample counts and the full unknown-texture list |
| `--debug-markers` | Wool marker at shapes that could not be placed (orange = missing mesh, red = parse failed, magenta = voxelize failed) |
| `--debug-transform` | Print the OPD → pivot → MC transform chain and place anchor markers |
| `--debug-grounding` | Print the grounding decision for every shape |
| `--debug-voxelize` | Print per-part voxelization diagnostics |
| `-v, --verbose` | Debug logging (unknown textures / unmapped GTD texture ids are logged at INFO) |

### Voxelization

`ko2mc/voxelizer.py` marks every voxel whose box overlaps a mesh triangle (13-axis separating-axis test), so thin walls, beams and roof edges are kept. `--voxel-mode` then chooses between stripping enclosed voxels (`surface`), keeping the raw hits (`hybrid`), or flood-filling enclosed air (`solid`). Billboard impostor parts (`_ip*`) are skipped; tree meshes (`_po*`) are split into log (lower 30 %) and leaves.

### Placement / audit report

`placement_report.json` contains, per shape: status (`placed`, `partial`, `missing_mesh`, `parse_failed`, `voxelize_failed`, `skipped`), KO and MC positions, grounding mode and reason, grounding quality (`support_ratio`, `buried_ratio`, `floating_ratio` from a 5x5 terrain sample grid over the footprint) and per-part resolver metadata and palette source. Top-level sections:

- `palette`: palette source counts (`texture`, `fallback_no_texture`, `fallback_unknown_texture`), `fallback_rate`, and the most frequent unknown texture names
- `terrain_palette`: GTD texture ids without a block mapping and their `fallback_rate`
- `grounding_summary`: mean support/buried/floating ratios and counts of mostly-floating / mostly-buried shapes
- `missing_refs`: mesh references no asset root could resolve

## Legacy converter (`ko2mc` / `python -m ko2mc`)

```bash
python -m ko2mc moradon.gtd --gtd-only                 # terrain only
python -m ko2mc moradon.gtd moradon.opd                # terrain + objects
python -m ko2mc -o ./worlds -n Moradon moradon.gtd moradon.opd
python -m ko2mc --scale 2 moradon.gtd moradon.opd      # 2 MC blocks per KO tile
```

| Argument | Description |
|----------|-------------|
| `gtd_file` | Path to `.gtd` terrain file (required) |
| `opd_file` | Path to `.opd` object file (optional) |
| `-o, --output` | Output directory (default: `./output`) |
| `-n, --name` | Minecraft world name |
| `-s, --scale` | Blocks per KO tile: 1, 2, or 4 (default: 1) |
| `--gtd-only` | Skip OPD object conversion |

### After conversion (both converters)

Copy the generated world folder to your Minecraft saves directory:

- **Windows**: `%appdata%/.minecraft/saves/`
- **Linux**: `~/.minecraft/saves/`
- **macOS**: `~/Library/Application Support/minecraft/saves/`

## How the legacy converter maps KO to Minecraft

### Terrain
Each KO tile (4m x 4m) becomes 1 Minecraft block (at scale=1). Heights are scaled so KO height 0 = MC Y=64. Below the surface, layers of dirt, stone, and bedrock are filled in.

### Texture mapping
KO texture IDs are mapped to Minecraft blocks: grass, dirt, sand, stone, gravel, cobblestone, snow, sandstone, clay, etc.

### Event objects
| KO Event | Minecraft Block |
|----------|----------------|
| Bind Point | Gold platform + Respawn Anchor |
| Gate | Iron Bars wall |
| Warp Gate | End Portal Frames |
| Barricade | Oak Fence wall |
| Magic Anvil | Anvil |
| Artifact | Iron Block + Beacon |
| Resurrection Point | Gold platform + Respawn Anchor |

### Structure objects
Object names are pattern-matched to place appropriate Minecraft structures:
- Trees → Oak Log + Leaves
- Rocks → Stone formations
- Buildings → Stone Brick shells
- Walls → Stone Brick Walls
- Lamps → Lanterns
- Water features → Water blocks

## Debug tools

```bash
python -m ko2mc.debug_mesh_info path/to/Object/ -r            # .n3pmesh statistics
python -m ko2mc.debug_voxelize_model model.n3pmesh -o /tmp/w  # voxelize one mesh into a test world
python verify_pivot.py [--json|--csv|--chain-check]           # pivot-fix verification report
```

## Tests

```bash
pip install -e ".[test]"   # or: pip install pytest numpy
python -m pytest -q
```

The tests use synthetic meshes/heightmaps and need no KO client data.

## Project structure

```
ko2mc/
  __main__.py            - legacy CLI entry point (ko2mc)
  converter.py           - legacy KO-to-MC conversion logic
  zone_converter.py      - OPD mesh voxelization pipeline + report (ko2mc-zone)
  voxelizer.py           - SAT triangle/voxel voxelizer, shell extraction, flood fill
  asset_resolver.py      - case-insensitive, priority-ordered asset lookup
  n3pmesh_parser.py      - .n3pmesh mesh parser
  smd_parser.py          - SMD mesh parser
  gtd_parser.py          - GTD terrain parser
  opd_parser.py          - OPD object parser (with decryption)
  math3d.py              - KO→MC coordinate, rotation, pivot and terrain-height math
  mc_world.py            - Minecraft Anvil world writer (NBT written directly)
  binary_reader.py       - binary reading helpers
  debug_*.py             - debug/inspection tools
tests/                   - pytest suite
verify_pivot.py          - pivot verification report
```

## Game data

This repository includes sample zone data: `gtd/` (terrain), `opd/` (object placement),
`tile/` and the original C++ `SMDExporter/` source. Client asset folders (meshes/textures,
e.g. from a 1886 or 1298 client) are not included — pass them with `--asset-roots`.

The actively developed copy of this converter lives in the
Knight-Online-Minecraft-Conversion-Plugin-Directory repository (`kol-mc-smd-generator-main/`);
see its `docs/ko2mc-voxelization-texture-audit.md` for the remediation plan behind the
grounding/palette/voxel-mode diagnostics.

## Credits

- Original SMD exporter by [Mustafa Kemal Gılor](https://github.com/mustafagilor)
- Minecraft conversion by this fork

## License

MIT

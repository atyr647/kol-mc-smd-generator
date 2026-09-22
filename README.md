# Knight Online to Minecraft Map Converter

Convert Knight Online `.gtd` (terrain) and `.opd` (object) map files into playable Minecraft Java Edition worlds, and preview both the original KO map and the result.

## What it does

- **Terrain**: reads the KO heightmap, the texture used on every tile, and the lakes/rivers, and builds solid Minecraft terrain with matching blocks and water.
- **Buildings, walls & trees**: built from the real KO 3D models (`.n3pmesh` + `.dxt` from the client's `Object` folder), with each block picked to match the KO texture colour there. Without the models, the server collision mesh in the `.opd` is used as a rough fallback.
- **Plants**: grass, flowers and reeds become blocks from their models too (or Minecraft plants without models / with `--simple-plants`).
- **Events**: warp gates, bind points, gates, anvils, etc. become recognizable Minecraft markers.
- **KO textures (optional)**: with the KO `.gtt` texture files, a resource pack paints the ground with the real KO textures.
- **Previews**: top-down maps and an interactive 3D viewer of the KO map, the Minecraft world, or both side by side.
- **Output**: a Minecraft Java Edition 1.20.4 world (tested by loading it in a real 1.20.4 server).

## Requirements

- Python 3.10+
- NumPy and Pillow

```bash
pip install -r requirements.txt
```

## Converting a map

```bash
# Moradon at true size (1 block = 1 meter). The .opd is found automatically.
python -m ko2mc gtd/moradon.gtd

# Smaller world: 1 block per 4 m tile
python -m ko2mc --scale 1 gtd/moradon.gtd

# Convert and make previews in one go
python -m ko2mc --preview gtd/moradon.gtd
```

| Argument | Description |
|----------|-------------|
| `gtd_file` | Path to `.gtd` terrain file (required) |
| `opd_file` | Path to `.opd` object file (optional, found in `opd/` automatically) |
| `-o, --output` | Output directory (default: `./output`) |
| `-n, --name` | Minecraft world name |
| `-s, --scale` | Blocks per 4 m KO tile: `4` = true size (default), `2`, `1` |
| `--vertical-scale` | Blocks per KO meter vertically (default: same as horizontal) |
| `--gtd-only` | Terrain only, ignore the `.opd` |
| `--no-objects` | Don't place trees, rocks, lamps, event markers |
| `--no-buildings` | Don't build walls/buildings from the collision mesh |
| `--ko-models DIR` | KO client `Object` folder (`.n3pmesh`, `.dxt`) for real buildings/trees (default: `object/` if it has any) |
| `--simple-plants` | With models: grass/flowers/reeds as single Minecraft plants |
| `--ko-textures DIR` | Folder with KO `.gtt` files (default: `dtex/` if it has any) |
| `--pack-resolution` | Pixels per block in the texture pack (default 32, KO's own detail) |
| `--pack-brightness` | Brightness multiplier for KO textures (default 1.3) |
| `--texture-map FILE` | JSON of extra `"texture regex": "block"` rules |
| `--preview` | Also render the previews |

Copy the world folder from `output/` into your Minecraft `saves` folder:

- **Windows**: `%appdata%/.minecraft/saves/`
- **Linux**: `~/.minecraft/saves/`
- **macOS**: `~/Library/Application Support/minecraft/saves/`

The converter prints where warp gates and bind points ended up, and how KO coordinates map to Minecraft ones.

## Real KO buildings and trees (3D models)

Every object in an `.opd` lists its model parts. With `--ko-models` pointing at the KO client's `Object` folder, each object is rebuilt from those models: parts are placed with the object's position, rotation and scale, the model surface is filled with blocks, and each block is the Minecraft block whose colour best matches the KO texture at that spot (leaves for green see-through parts like tree leaves). Every model is converted, including grass, flowers and reeds; add `--simple-plants` to use single Minecraft plants for those instead. Effect objects (glows, smoke, sparkles) are skipped because they aren't solid.

```bash
C="../knight-online-minecraft-conversion-plugin-directory/additions/USKO Moradon Patch (v1298)/Client"
python -m ko2mc gtd/moradon.gtd --ko-models "$C/Object" --ko-textures "$C/DTex"
```

The preview then also draws the KO side with the real textured models and ground textures, so the split-screen viewer compares like with like. With the full map this page is large (about 30 MB for Moradon); use `--area x1,z1,x2,z2` to focus on one spot.

## Real KO ground textures (resource pack)

Put the `.gtt` files from your KO client's `Data/dtex` folder into `dtex/` (see `dtex/README.md`), or point `--ko-textures` at a folder that has them, and convert again. For example, with the `Knight-Online-Minecraft-Conversion-Plugin-Directory` repo cloned next to this one:

```bash
python -m ko2mc gtd/moradon.gtd --ko-textures "../knight-online-minecraft-conversion-plugin-directory/additions/USKO Moradon Patch (v1298)/Client/DTex"
```

That folder covers 43 of the 48 maps here. `In_dungeon06`, `dungeon_a`, `dungeon_defense`, `eslantzone` and `war_a` also use newer (2013–2017) textures; tiles whose file is missing keep a normal Minecraft block. Use `--pack-brightness` (default 1.3) if the ground looks too dark or too bright.

You get:

- `output/KO_Moradon/resources.zip`: Minecraft applies it automatically when you open that world in singleplayer.
- `output/KO_Moradon_KO_textures.zip`: a copy for your `resourcepacks` folder (for servers or other setups).

How it works:

- The ground is rebuilt the way the KO client draws it: each 4 m tile has a base texture plus, on transition tiles, an overlay texture that is added on top (that's how grass fades into dirt), and each layer can be flipped or rotated.
- Textures keep KO's scale: at true size one KO tile covers 4 x 4 blocks, so every block shows its own 32 x 32 pixel piece of the tile.
- A map has thousands of different pieces, but a resource pack can only add about 800 new block looks (as *note block states*: instrument + note). Similar pieces are grouped and each group uses the real KO piece closest to the group's average.
- Minecraft derives a note block's instrument from the block beneath it, so the matching block (stone, sand, wool...) is placed right under each textured block. That way the ground keeps its texture when you build next to it. Right-clicking a textured block changes its note, so it's best for exploring rather than survival.

## Previews

```bash
# Original KO map (reference)
python -m ko2mc.preview ko gtd/moradon.gtd

# The converted world, read back from the world files like Minecraft reads them
python -m ko2mc.preview mc output/KO_Moradon

# Both, lined up, plus a side-by-side image and split-screen 3D viewer
python -m ko2mc.preview compare output/KO_Moradon
```

Files are written to `output/KO_Moradon/preview/` (or `preview/` for `ko`):

- `*_map.png`: top-down map, north up, 1 pixel = 1 Minecraft block.
- `compare_map.png`: KO | Minecraft side by side.
- `*_3d.html`: open in a browser (needs internet to load three.js). Drag to rotate, right-drag to pan, wheel to zoom; `F` = fly like in creative mode (WASD, Space, Shift); `1`/`2`/`3` = KO / Minecraft / split screen; "Player view" puts you at the spawn at eye height.

To see real Minecraft textures in the preview, add `--download-textures` (downloads the official 1.20.4 client jar from Mojang once, into `~/.cache/ko2mc`) or `--mc-jar path/to/1.20.4.jar`. An installed Minecraft is found automatically. Without a jar, plain colours are used. The KO texture pack is shown automatically when the world has one.

## How it maps KO to Minecraft

- KO uses meters with z pointing **north**; Minecraft z points **south**, so the map is flipped on z (otherwise it would be mirrored). With the default scale: `mc_x = ko_x`, `mc_z = map_size - ko_z`.
- Heights keep their real proportions. KO height 0 is Y=64 when the map fits; very tall maps are shifted or squashed to fit Minecraft's height limit (the converter tells you).
- Terrain blocks come from the tile texture names (`grass` → grass block, `brick` → stone bricks, `snow` → snow, `ground` → coarse dirt...). Edit `ko2mc/materials.py` or use `--texture-map` to change them.

## Project structure

```
ko2mc/
  __main__.py     - converter command line
  gtd_parser.py   - .gtd terrain parser (heights, tile textures, water)
  opd_parser.py   - .opd parser (objects, collision mesh, events)
  materials.py    - texture name -> block rules
  converter.py    - KO -> Minecraft conversion
  mc_world.py     - Minecraft world writer (region files, level.dat)
  ko_textures.py  - .gtt/.dxt texture reader and resource pack writer
  ko_models.py    - .n3pmesh model reader and model -> blocks
  nbt.py          - NBT reader/writer
  mca_reader.py   - reads worlds back for previews
  mc_textures.py  - Minecraft block textures for previews
  preview.py      - preview command line
  viewer.html     - 3D viewer template
SMDExporter/      - original C++ server map (.gsmd) generator
```

## Sample data

`gtd/`, `opd/` and `tile/` contain Knight Online map files: Moradon, El Morad, Karus, battle zones, dungeons, and more.

Note for the C++ `SMDExporter`: the `.gtd` heightmap is stored x-major (`index = x * N + z`). `CGameTerrain::LoadFromStream` reads it z-major, which transposes the heights.

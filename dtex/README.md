# KO terrain textures go here

Copy the `.gtt` files from your Knight Online client's `Data/dtex` folder into
this folder (e.g. `map_mora_brick01_0.gtt`, `map_el_ngrass01_1.gtt`).

When this folder contains `.gtt` files, `python -m ko2mc ...` automatically builds
a Minecraft resource pack with the real KO ground textures. To see which files a
map needs, run the converter; it lists any missing ones.

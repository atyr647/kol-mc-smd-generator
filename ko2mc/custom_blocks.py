"""Which Minecraft block states get a KO texture through the resource pack.

A vanilla game client can't be sent new block types, so (like the ItemsAdder /
Oraxen server plugins) we give *existing* block states new models and textures
in the resource pack. These "host" states are chosen so that nothing else in a
converted world uses them:

  ground   note block states (instrument + note); the block underneath keeps the
           instrument stable (see ko_textures.py)
  solid    ~500 full-cube states: rarely used decorative blocks (glazed
           terracotta, mushroom blocks, wool, ores, stone variants...) plus the
           mob-head note block instruments
  foliage  leaf states (see-through like leaves, but with KO leaf textures)
  plant    tripwire states drawn as crossed plant sprites (KO grass and flowers)
  stairs / slab   stairs and slab block types re-textured with KO step textures

Retexturing affects those blocks everywhere in a world using the pack, e.g. wool
placed by a player later also shows a KO texture. The companion server plugin
(plugin/ko2mc-blocks) stops Minecraft from changing these states on its own
(leaf distance, note block instrument, tripwire connections...).
"""

import itertools

COLORS = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
          "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]

# Single-state full cubes that a converted world never places itself.
SINGLE_HOSTS = (
    [f"{c}_wool" for c in COLORS if c != "white"]             # white wool: note block instrument
    + [f"{c}_concrete" for c in COLORS]
    + [f"{c}_terracotta" for c in COLORS] + ["terracotta"]
    + ["coal_ore", "iron_ore", "copper_ore", "gold_ore", "lapis_ore", "diamond_ore", "emerald_ore",
       "deepslate_coal_ore", "deepslate_iron_ore", "deepslate_copper_ore", "deepslate_gold_ore",
       "deepslate_lapis_ore", "deepslate_diamond_ore", "deepslate_emerald_ore",
       "nether_gold_ore", "nether_quartz_ore",
       "raw_iron_block", "raw_copper_block", "raw_gold_block", "coal_block", "diamond_block",
       "lapis_block", "netherite_block", "amethyst_block",
       "bricks", "mud_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "chiseled_stone_bricks",
       "mossy_cobblestone", "deepslate_bricks", "cracked_deepslate_bricks", "deepslate_tiles",
       "cracked_deepslate_tiles", "chiseled_deepslate", "polished_deepslate", "cobbled_deepslate",
       "nether_bricks", "cracked_nether_bricks", "chiseled_nether_bricks", "red_nether_bricks",
       "end_stone_bricks", "end_stone", "purpur_block", "prismarine", "prismarine_bricks",
       "dark_prismarine", "quartz_block", "quartz_bricks", "chiseled_quartz_block", "smooth_quartz",
       "blackstone", "polished_blackstone", "polished_blackstone_bricks",
       "cracked_polished_blackstone_bricks", "chiseled_polished_blackstone", "gilded_blackstone",
       "granite", "polished_granite", "diorite", "polished_diorite", "andesite", "smooth_stone",
       "smooth_basalt", "dripstone_block", "packed_mud", "chiseled_sandstone", "cut_sandstone",
       "smooth_sandstone", "red_sandstone", "cut_red_sandstone", "chiseled_red_sandstone",
       "smooth_red_sandstone", "obsidian", "sponge", "dried_kelp_block", "honeycomb_block",
       "nether_wart_block", "warped_wart_block", "lodestone", "ancient_debris", "reinforced_deepslate",
       "spruce_planks", "birch_planks", "jungle_planks", "acacia_planks", "dark_oak_planks",
       "mangrove_planks", "cherry_planks", "bamboo_planks", "bamboo_mosaic", "crimson_planks",
       "warped_planks", "waxed_copper_block", "waxed_exposed_copper", "waxed_weathered_copper",
       "waxed_oxidized_copper", "waxed_cut_copper", "waxed_exposed_cut_copper",
       "waxed_weathered_cut_copper", "waxed_oxidized_cut_copper"]
)
AXIS_HOSTS = ([f"{w}_wood" for w in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry")]
              + [f"stripped_{w}_wood" for w in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry")]
              + ["crimson_hyphae", "warped_hyphae", "stripped_crimson_hyphae", "stripped_warped_hyphae",
                 "basalt", "polished_basalt", "quartz_pillar", "purpur_pillar", "muddy_mangrove_roots"])
GLAZED_HOSTS = [f"{c}_glazed_terracotta" for c in COLORS]
MUSHROOM_HOSTS = ["brown_mushroom_block", "red_mushroom_block", "mushroom_stem"]
MOB_INSTRUMENTS = ["zombie", "skeleton", "creeper", "dragon", "wither_skeleton", "piglin", "custom_head"]

LEAF_HOSTS = ["oak_leaves", "spruce_leaves", "birch_leaves", "jungle_leaves", "acacia_leaves",
              "dark_oak_leaves", "mangrove_leaves", "cherry_leaves", "azalea_leaves", "flowering_azalea_leaves"]

STAIR_SLAB_TYPES = [
    "oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "bamboo",
    "bamboo_mosaic", "crimson", "warped", "stone", "cobblestone", "mossy_cobblestone", "stone_brick",
    "mossy_stone_brick", "granite", "polished_granite", "diorite", "polished_diorite", "andesite",
    "polished_andesite", "cobbled_deepslate", "polished_deepslate", "deepslate_brick", "deepslate_tile",
    "brick", "mud_brick", "sandstone", "smooth_sandstone", "red_sandstone", "smooth_red_sandstone",
    "nether_brick", "red_nether_brick", "quartz", "smooth_quartz", "purpur", "prismarine",
    "prismarine_brick", "dark_prismarine", "blackstone", "polished_blackstone",
    "polished_blackstone_brick", "end_stone_brick",
]


def _combos(props: dict) -> list[dict]:
    keys = list(props)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(props[k] for k in keys))]


def _key(d: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(d.items()))


def _state(block: str, d: dict) -> str:
    return f"minecraft:{block}" + (f"[{_key(d)}]" if d else "")


def solid_slots():
    """(world block state, host block, blockstate variant keys) for every solid slot, best first."""
    out = []
    for b in SINGLE_HOSTS:
        out.append((_state(b, {}), b, [""]))
    for b in GLAZED_HOSTS:
        for f in ("north", "south", "east", "west"):
            out.append((_state(b, {"facing": f}), b, [f"facing={f}"]))
    for b in AXIS_HOSTS:
        for a in ("x", "y", "z"):
            out.append((_state(b, {"axis": a}), b, [f"axis={a}"]))
    for b in MUSHROOM_HOSTS:
        for d in _combos({s: ("true", "false") for s in ("down", "east", "north", "south", "up", "west")}):
            out.append((_state(b, d), b, [_key(d)]))
    for inst in MOB_INSTRUMENTS:
        for note in range(25):
            ks = [f"instrument={inst},note={note},powered={p}" for p in ("false", "true")]
            out.append((f"minecraft:note_block[instrument={inst},note={note},powered=false]", "note_block", ks))
    return out


def foliage_slots():
    out = []
    for b in LEAF_HOSTS:
        for dist in range(1, 8):
            ks = [f"distance={dist},persistent={p},waterlogged={w}" for p in ("false", "true") for w in ("false", "true")]
            out.append((f"minecraft:{b}[distance={dist},persistent=true,waterlogged=false]", b, ks))
    return out


def plant_slots():
    out = []
    for d in _combos({s: ("false", "true") for s in ("attached", "disarmed", "east", "north", "south", "west")}):
        ks = [_key({**d, "powered": p}) for p in ("false", "true")]
        out.append((_state("tripwire", {**d, "powered": "false"}), "tripwire", ks))
    return out


def host_blocks() -> set[str]:
    """Every block whose look is replaced (for the server plugin)."""
    return (set(SINGLE_HOSTS) | set(GLAZED_HOSTS) | set(AXIS_HOSTS) | set(MUSHROOM_HOSTS)
            | set(LEAF_HOSTS) | {"note_block", "tripwire"}
            | {f"{t}_stairs" for t in STAIR_SLAB_TYPES} | {f"{t}_slab" for t in STAIR_SLAB_TYPES})


def all_block_states(host: str) -> list[str]:
    """All blockstate variant keys of a host block (for a complete blockstates file)."""
    if host in SINGLE_HOSTS:
        return [""]
    if host in GLAZED_HOSTS:
        return [f"facing={f}" for f in ("north", "south", "east", "west")]
    if host in AXIS_HOSTS:
        return [f"axis={a}" for a in ("x", "y", "z")]
    if host in MUSHROOM_HOSTS:
        return [_key(d) for d in _combos({s: ("true", "false") for s in ("down", "east", "north", "south", "up", "west")})]
    if host in LEAF_HOSTS:
        return [f"distance={d},persistent={p},waterlogged={w}" for d in range(1, 8)
                for p in ("false", "true") for w in ("false", "true")]
    if host == "tripwire":
        return [_key(d) for d in _combos({s: ("false", "true") for s in
                                           ("attached", "disarmed", "east", "north", "powered", "south", "west")})]
    raise KeyError(host)

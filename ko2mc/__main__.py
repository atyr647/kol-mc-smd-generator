"""CLI entry point for the Knight Online to Minecraft converter."""

import argparse
import os
import sys

from . import materials
from .converter import convert_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert Knight Online .gtd/.opd map files to Minecraft worlds.",
        epilog=(
            "Examples:\n"
            "  %(prog)s gtd/moradon.gtd opd/moradon.opd\n"
            "  %(prog)s --scale 1 --name Moradon gtd/moradon.gtd opd/moradon.opd\n"
            "  %(prog)s --gtd-only gtd/arena.gtd\n"
            "  %(prog)s --preview gtd/moradon.gtd opd/moradon.opd\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("gtd_file", help="Path to .gtd (Game Terrain Data) file")
    parser.add_argument("opd_file", nargs="?", default=None,
                        help="Path to .opd (Object Post Data) file (optional; "
                             "looked up in ./opd automatically)")
    parser.add_argument("-o", "--output", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("-n", "--name", default=None,
                        help="World name (default: derived from GTD filename)")
    parser.add_argument("-s", "--scale", type=int, default=4, choices=[1, 2, 4],
                        help="Blocks per KO tile (4 m): 4 = true size, 1 block per meter "
                             "(default); 2 = half size; 1 = quarter size")
    parser.add_argument("--vertical-scale", type=float, default=None,
                        help="Blocks per KO meter vertically (default: same as horizontal, "
                             "so hills keep their real shape)")
    parser.add_argument("--gtd-only", action="store_true",
                        help="Only convert terrain from GTD (ignore OPD)")
    parser.add_argument("--no-objects", action="store_true",
                        help="Don't place trees, rocks, lamps and event markers")
    parser.add_argument("--no-buildings", action="store_true",
                        help="Don't build walls/buildings from the collision mesh")
    parser.add_argument("--texture-map", default=None,
                        help="JSON file with extra texture-name -> block rules")
    parser.add_argument("--ko-textures", default=None, metavar="DIR",
                        help="Folder with KO .gtt terrain textures (client Data/dtex); "
                             "makes a resource pack with the real KO ground textures "
                             "(default: ./dtex if it exists)")
    parser.add_argument("--pack-resolution", type=int, default=64, choices=[16, 32, 64, 128, 256],
                        help="Pixels per block texture in the resource pack (default 64)")
    parser.add_argument("--preview", action="store_true",
                        help="Also render the KO reference and Minecraft previews")

    args = parser.parse_args()

    if not os.path.exists(args.gtd_file):
        print(f"Error: GTD file not found: {args.gtd_file}", file=sys.stderr)
        sys.exit(1)

    if args.ko_textures is None:
        # Use a dtex/ folder next to gtd/ automatically (where the .gtt files go)
        guess = os.path.join(os.path.dirname(os.path.abspath(args.gtd_file)), "..", "dtex")
        for folder in (guess, "dtex"):
            if os.path.isdir(folder) and any(f.lower().endswith(".gtt") for f in os.listdir(folder)):
                args.ko_textures = os.path.normpath(folder)
                break

    if args.texture_map:
        materials.load_texture_map(args.texture_map)

    opd_path = None
    if not args.gtd_only:
        opd_path = args.opd_file or find_opd(args.gtd_file)
        if opd_path and not os.path.exists(opd_path):
            print(f"Warning: OPD file not found: {opd_path}", file=sys.stderr)
            print("Continuing with terrain only...", file=sys.stderr)
            opd_path = None

    world_name = args.name
    if not world_name:
        world_name = os.path.splitext(os.path.basename(args.gtd_file))[0]
        world_name = world_name.replace("_", " ").title().replace(" ", "")
        world_name = f"KO_{world_name}"

    world_dir = convert_map(
        gtd_path=args.gtd_file,
        opd_path=opd_path,
        output_dir=args.output,
        world_name=world_name,
        scale=args.scale,
        vertical_scale=args.vertical_scale,
        objects=not args.no_objects,
        buildings=not args.no_buildings,
        ko_textures=args.ko_textures,
        pack_resolution=args.pack_resolution,
    )

    if args.preview:
        from .preview import preview_compare
        preview_compare(world_dir)


def find_opd(gtd_path: str) -> str | None:
    """Find the .opd that belongs to a .gtd (same name, next to it or in ../opd)."""
    base = os.path.splitext(os.path.basename(gtd_path))[0].lower()
    here = os.path.dirname(os.path.abspath(gtd_path))
    for folder in (here, os.path.join(here, "..", "opd"), "opd"):
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower() == base + ".opd":
                    return os.path.join(folder, f)
    return None


if __name__ == "__main__":
    main()

"""
Merge Glass Objects (Copos) into Scene
=======================================
Import a `Copos` collection of glass cup objects from an external .blend
asset pack into the currently open .blend file.

Usage (run inside Blender):
    blender your_base.blend --python merge_glass_objects.py -- \
        --copos /path/to/Copos.blend \
        [--exclude name1,name2,...]

If --copos is omitted, the script aborts with a usage hint. Use --exclude to
drop heavy or non-converging meshes after import (object names are matched
exactly, case-sensitive). Comma-separated; surround with quotes if any name
contains a comma.
"""

import bpy
import os
import sys
import argparse


def parse_args():
    """Parse args after Blender's `--` separator."""
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    parser = argparse.ArgumentParser(
        description="Import the Copos glass-cup collection from an external .blend.",
    )
    parser.add_argument(
        '--copos',
        type=str,
        default=os.environ.get('COPOS_BLEND', None),
        help="Path to the source .blend file containing a Copos collection. "
             "Falls back to the COPOS_BLEND environment variable.",
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default='',
        help="Comma-separated object names to drop after import "
             "(e.g. heavy meshes whose caustics don't converge).",
    )
    return parser.parse_args(argv)


def merge_copos(copos_path: str, exclude: list):
    print("\n" + "=" * 60)
    print("Importing glass collection (Copos)")
    print("=" * 60)

    if not copos_path:
        print("\nError: no source .blend specified.")
        print("Usage: blender base.blend --python merge_glass_objects.py -- "
              "--copos /path/to/Copos.blend")
        sys.exit(1)

    if not os.path.exists(copos_path):
        print(f"\nError: file not found: {copos_path}")
        sys.exit(1)

    print(f"\nLoading from: {copos_path}")

    with bpy.data.libraries.load(copos_path, link=False) as (data_from, data_to):
        if 'Copos' in data_from.collections:
            data_to.collections = ['Copos']
            print("  Found 'Copos' collection")
        else:
            print("  No 'Copos' collection in source — importing all objects")
            data_to.objects = data_from.objects

    if 'Copos' in bpy.data.collections:
        copos_col = bpy.data.collections['Copos']
        if copos_col.name not in [c.name for c in bpy.context.scene.collection.children]:
            bpy.context.scene.collection.children.link(copos_col)
            print("  Linked 'Copos' collection into the scene")
    else:
        copos_col = bpy.data.collections.new('Copos')
        bpy.context.scene.collection.children.link(copos_col)
        for obj in data_to.objects:
            if obj is not None:
                copos_col.objects.link(obj)
        print("  Created 'Copos' collection from imported objects")

    glass_objects = []
    if 'Copos' in bpy.data.collections:
        glass_objects = [o.name for o in bpy.data.collections['Copos'].objects]

    print(f"\nImported glass objects: {len(glass_objects)}")
    for name in glass_objects:
        status = "[exclude]" if name in exclude else "[ok]"
        verts = 0
        obj = bpy.data.objects.get(name)
        if obj and obj.type == 'MESH':
            verts = len(obj.data.vertices)
        print(f"  {status} {name} ({verts} verts)")

    if exclude:
        print("\nRemoving excluded objects...")
        removed = 0
        for name in exclude:
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
                print(f"  removed: {name}")
                removed += 1
        print(f"  total removed: {removed}")

    print("\n" + "=" * 60)
    print("Done. Save the .blend (Ctrl+S) to keep the changes.")
    print("=" * 60)

    if 'Copos' in bpy.data.collections:
        remaining = [o.name for o in bpy.data.collections['Copos'].objects]
        print(f"\n'Copos' collection now has {len(remaining)} object(s):")
        for name in remaining:
            obj = bpy.data.objects.get(name)
            if obj and obj.type == 'MESH':
                print(f"  - {name} ({len(obj.data.vertices)} verts)")


if __name__ == "__main__":
    args = parse_args()
    exclude = [n.strip() for n in args.exclude.split(',') if n.strip()]
    merge_copos(args.copos, exclude)

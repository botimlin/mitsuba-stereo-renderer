"""
Merge a glass-cup collection from an external .blend asset pack into the
currently open .blend file.

The randomizer expects the imported collection to be named `Glass` (the
contract checked by `check_blend_objects.py`). If your source .blend stores
its glass cups under a different collection name (commonly `Copos`,
`Cups`, etc.), pass `--source-collection`.

Usage (run inside Blender):
    blender your_base.blend --python merge_glass_objects.py -- \
        --src /path/to/glass_pack.blend \
        [--source-collection Copos] \
        [--target-collection Glass] \
        [--exclude name1,name2,...]

If --src is omitted the script aborts with a usage hint. Use --exclude to
drop heavy or non-converging meshes after import (object names matched
exactly, case-sensitive, comma-separated).
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
        description="Import a glass-cup collection from an external .blend.",
    )
    parser.add_argument(
        '--src',
        type=str,
        default=os.environ.get('GLASS_BLEND', None),
        help="Path to the source .blend file. Falls back to GLASS_BLEND env var.",
    )
    parser.add_argument(
        '--source-collection',
        type=str,
        default='Glass',
        help="Collection name to look for inside the source .blend "
             "(default: 'Glass'; some packs use 'Copos', 'Cups', etc.).",
    )
    parser.add_argument(
        '--target-collection',
        type=str,
        default='Glass',
        help="Collection name to use in the current .blend after import "
             "(default: 'Glass' — matches the randomizer contract).",
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default='',
        help="Comma-separated object names to drop after import "
             "(e.g. heavy meshes whose caustics don't converge).",
    )
    return parser.parse_args(argv)


def merge_glass(src_path: str, source_col: str, target_col: str, exclude: list):
    print("\n" + "=" * 60)
    print(f"Importing glass collection: '{source_col}' -> '{target_col}'")
    print("=" * 60)

    if not src_path:
        print("\nError: no source .blend specified.")
        print("Usage: blender base.blend --python merge_glass_objects.py -- "
              "--src /path/to/glass_pack.blend")
        sys.exit(1)

    if not os.path.exists(src_path):
        print(f"\nError: file not found: {src_path}")
        sys.exit(1)

    print(f"\nLoading from: {src_path}")

    with bpy.data.libraries.load(src_path, link=False) as (data_from, data_to):
        if source_col in data_from.collections:
            data_to.collections = [source_col]
            print(f"  Found '{source_col}' collection")
        else:
            print(f"  No '{source_col}' collection in source — importing all objects")
            data_to.objects = data_from.objects

    # Rename the imported collection to the target name (if different)
    if source_col != target_col and source_col in bpy.data.collections:
        bpy.data.collections[source_col].name = target_col
        print(f"  Renamed '{source_col}' -> '{target_col}'")

    if target_col in bpy.data.collections:
        col = bpy.data.collections[target_col]
        if col.name not in [c.name for c in bpy.context.scene.collection.children]:
            bpy.context.scene.collection.children.link(col)
            print(f"  Linked '{target_col}' collection into the scene")
    else:
        col = bpy.data.collections.new(target_col)
        bpy.context.scene.collection.children.link(col)
        for obj in data_to.objects:
            if obj is not None:
                col.objects.link(obj)
        print(f"  Created '{target_col}' collection from imported objects")

    glass_objects = [o.name for o in bpy.data.collections[target_col].objects]

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

    remaining = [o.name for o in bpy.data.collections[target_col].objects]
    print(f"\n'{target_col}' collection now has {len(remaining)} object(s):")
    for name in remaining:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == 'MESH':
            print(f"  - {name} ({len(obj.data.vertices)} verts)")


if __name__ == "__main__":
    args = parse_args()
    exclude = [n.strip() for n in args.exclude.split(',') if n.strip()]
    merge_glass(args.src, args.source_collection, args.target_collection, exclude)

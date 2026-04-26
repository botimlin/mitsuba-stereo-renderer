"""
Inspect Blender file objects and collections.
Run from the Blender Scripting tab.

Verifies whether the currently open .blend follows the naming contract that
`blender_glass_randomizer.py` expects:

    Required collections
    --------------------
    Glass      - glass cup mesh objects (renderer treats these as transparent)
    Tables     - candidate table mesh objects (optional if you set
                 `table.source_names` directly in the randomizer CONFIG)
    Furniture  - floor furniture (chairs, stools, lamps, etc.)
    Cabinets   - large cabinets / shelves (rotation-locked placement)
    Decor      - tabletop decorations (vases, small props)

If any collection is missing or empty the randomizer falls back gracefully —
that category simply produces no items. This script tells you which
collections are present and how many mesh objects each one contains.
"""

import bpy

REQUIRED_COLLECTIONS = [
    'Glass',
    'Tables',
    'Furniture',
    'Cabinets',
    'Decor',
]

print("\n" + "=" * 60)
print("Blender object & collection inspector")
print("=" * 60)

# All collections
print("\n=== All Collections ===")
for col in bpy.data.collections:
    mesh_count = len([o for o in col.objects if o.type == 'MESH'])
    print(f"  {col.name}: {mesh_count} mesh objects")
    for obj in col.objects:
        if obj.type == 'MESH':
            d = obj.dimensions
            print(f"    - {obj.name} ({d.x*1000:.0f}x{d.y*1000:.0f}x{d.z*1000:.0f}mm)")

# Required collections check
print("\n=== Randomizer contract check ===")
for col_name in REQUIRED_COLLECTIONS:
    col = bpy.data.collections.get(col_name)
    if col is None:
        print(f"  [missing] {col_name}")
        continue
    mesh_objs = [o for o in col.objects if o.type == 'MESH']
    print(f"  [ok]      {col_name}: {len(mesh_objs)} mesh objects")
    for obj in mesh_objs[:5]:
        print(f"              - {obj.name}")
    if len(mesh_objs) > 5:
        print(f"              ... and {len(mesh_objs) - 5} more")

print("\n" + "=" * 60)

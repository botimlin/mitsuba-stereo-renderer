"""
Inspect Blender file objects and collections.
Run from the Blender Scripting tab.

Verifies whether the currently open .blend follows the naming contract that
`blender_glass_randomizer.py` expects:

    Required collections
    --------------------
    Copos          - glass cup mesh objects
    SB_Tables      - candidate table mesh objects (optional if `table.source_names` is set)
    SB_Furniture   - floor furniture (chairs, lamps, etc.)
    SB_Cabinets    - large cabinets / shelves
    SB_Decor       - tabletop decorations (vases, etc.)
    SB_Vases       - reserved (alternative decor grouping)
    SB_Lamps       - reserved
    SB_Plants      - reserved

If any of these are missing the randomizer falls back gracefully (no items
of that category get placed). This script tells you which collections are
present and how many mesh objects they contain.
"""

import bpy

REQUIRED_COLLECTIONS = [
    'Copos',
    'SB_Tables',
    'SB_Furniture',
    'SB_Cabinets',
    'SB_Decor',
    'SB_Vases',
    'SB_Lamps',
    'SB_Plants',
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

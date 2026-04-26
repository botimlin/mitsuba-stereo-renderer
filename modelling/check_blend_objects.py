"""
檢查 Blender 檔案中的物件和 Collections
在 Blender 中執行此腳本
"""

import bpy

print("\n" + "=" * 60)
print("Blender 物件檢查")
print("=" * 60)

# 檢查 Collections
print("\n=== Collections ===")
for col in bpy.data.collections:
    obj_count = len([o for o in col.objects if o.type == 'MESH'])
    print(f"  {col.name}: {obj_count} mesh objects")
    for obj in col.objects:
        if obj.type == 'MESH':
            dims = obj.dimensions
            print(f"    - {obj.name} ({dims.x*1000:.0f}x{dims.y*1000:.0f}x{dims.z*1000:.0f}mm)")

# 檢查需要的桌子物件
print("\n=== 桌子物件 (VintageTable*) ===")
table_names = ['VintageTable', 'VintageTable2', 'VintageTable3', 'Table']
for name in table_names:
    obj = bpy.data.objects.get(name)
    if obj:
        dims = obj.dimensions
        print(f"  ✓ {name}: {dims.x*1000:.0f}x{dims.y*1000:.0f}x{dims.z*1000:.0f}mm")
    else:
        print(f"  ✗ {name}: 不存在")

# 檢查 SB_* Collections
print("\n=== SB_* Collections ===")
sb_collections = ['SB_Vases', 'SB_Lamps', 'SB_Plants', 'SB_Decor', 'SB_Furniture']
for col_name in sb_collections:
    col = bpy.data.collections.get(col_name)
    if col:
        mesh_objs = [o for o in col.objects if o.type == 'MESH']
        print(f"  ✓ {col_name}: {len(mesh_objs)} objects")
        for obj in mesh_objs[:5]:  # 只顯示前5個
            print(f"      - {obj.name}")
        if len(mesh_objs) > 5:
            print(f"      ... 還有 {len(mesh_objs) - 5} 個")
    else:
        print(f"  ✗ {col_name}: 不存在")

# 檢查 Copos Collection (玻璃)
print("\n=== Copos Collection (玻璃) ===")
copos = bpy.data.collections.get('Copos')
if copos:
    mesh_objs = [o for o in copos.objects if o.type == 'MESH']
    print(f"  ✓ Copos: {len(mesh_objs)} objects")
    for obj in mesh_objs:
        dims = obj.dimensions
        print(f"      - {obj.name} ({dims.x*1000:.0f}x{dims.y*1000:.0f}x{dims.z*1000:.0f}mm)")
else:
    print(f"  ✗ Copos: 不存在")

print("\n" + "=" * 60)

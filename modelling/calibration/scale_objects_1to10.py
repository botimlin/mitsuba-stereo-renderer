"""
將所有物件縮放到 1:10 比例
在 Blender 中執行此腳本
"""

import bpy
from mathutils import Vector

SCALE_FACTOR = 0.1  # 1:10 比例

# 要縮放的 Collections
COLLECTIONS_TO_SCALE = [
    'Glass',
    'Tables',
    'Furniture',
    'Cabinets',
    'Decor',
]

print("\n" + "=" * 60)
print(f"縮放所有物件到 1:{int(1/SCALE_FACTOR)} 比例")
print("=" * 60)

scaled_count = 0

for col_name in COLLECTIONS_TO_SCALE:
    col = bpy.data.collections.get(col_name)
    if col is None:
        print(f"\n[!] Collection '{col_name}' 不存在，跳過")
        continue

    print(f"\n=== {col_name} ===")

    for obj in col.objects:
        if obj.type != 'MESH':
            continue

        # 記錄原始尺寸
        old_dims = obj.dimensions.copy()

        # 取消選擇所有物件
        bpy.ops.object.select_all(action='DESELECT')

        # 選擇並激活此物件
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # 縮放
        obj.scale = (SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)

        # Apply scale
        bpy.ops.object.transform_apply(scale=True)

        # 設定原點到底部中心
        # 先把 cursor 移到物件底部
        bpy.context.view_layer.update()
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        min_z = min(c.z for c in corners)
        center_x = sum(c.x for c in corners) / 8
        center_y = sum(c.y for c in corners) / 8

        # 移動 cursor 到底部中心
        bpy.context.scene.cursor.location = (center_x, center_y, min_z)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

        # 把物件移到原點
        obj.location = (0, 0, 0)

        # 記錄新尺寸
        new_dims = obj.dimensions

        print(f"  {obj.name}:")
        print(f"    {old_dims.x*1000:.0f}x{old_dims.y*1000:.0f}x{old_dims.z*1000:.0f}mm -> "
              f"{new_dims.x*1000:.0f}x{new_dims.y*1000:.0f}x{new_dims.z*1000:.0f}mm")

        scaled_count += 1

# 重置 cursor
bpy.context.scene.cursor.location = (0, 0, 0)

print("\n" + "=" * 60)
print(f"完成！共縮放 {scaled_count} 個物件")
print("=" * 60)
print("\n記得存檔！(Ctrl+S 或 File > Save)")

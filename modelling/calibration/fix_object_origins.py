"""
Fix Object Origins and Transforms
==================================
整理物件的原點和座標，讓物件能正確放置在桌面上

使用方式：在 Blender Scripting 分頁執行此腳本
"""

import bpy

# ============================================================
# 配置
# ============================================================

# 要整理的 Collections
COLLECTIONS_TO_FIX = ['Glass', 'Tables', 'Furniture', 'Cabinets', 'Decor']

# ============================================================
# 主要函數
# ============================================================

def fix_object_transforms():
    print("\n" + "=" * 60)
    print("整理物件原點和座標")
    print("=" * 60)

    fixed_count = 0

    # 先取消所有選擇
    bpy.ops.object.select_all(action='DESELECT')

    for col_name in COLLECTIONS_TO_FIX:
        if col_name not in bpy.data.collections:
            print(f"\n⚠️ 找不到 Collection: {col_name}")
            continue

        col = bpy.data.collections[col_name]
        print(f"\n📁 {col_name}")

        for obj in col.objects:
            if obj.type != 'MESH':
                continue

            try:
                # 確保物件可見且可選
                obj.hide_set(False)
                obj.hide_viewport = False

                # 選擇物件
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)

                # 1. Apply rotation & scale (保留 location)
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

                # 2. 原點設到 geometry 中心 (bounds)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

                # 3. 計算物件底部到原點的距離，調整讓底部在 Z=0
                # 目前原點在 bounds 中心，需要把原點移到底部
                half_height = obj.dimensions.z / 2

                # 移動物件讓底部在 Z=0
                obj.location.z = half_height

                # Apply location
                bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

                # 重新設定原點到底部
                # 先把 3D cursor 移到物件底部中心
                cursor_backup = bpy.context.scene.cursor.location.copy()
                bpy.context.scene.cursor.location = (0, 0, 0)
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
                bpy.context.scene.cursor.location = cursor_backup

                # 最後把物件位置歸零
                obj.location = (0, 0, 0)

                obj.select_set(False)

                verts = len(obj.data.vertices)
                dims = obj.dimensions
                print(f"   ✅ {obj.name} ({verts} verts, {dims.x:.1f}×{dims.y:.1f}×{dims.z:.1f} mm)")
                fixed_count += 1

            except Exception as e:
                print(f"   ❌ {obj.name}: {e}")
                obj.select_set(False)

    print("\n" + "=" * 60)
    print(f"完成！共整理 {fixed_count} 個物件")
    print("=" * 60)
    print("\n物件現在：")
    print("  • 原點在底部中心")
    print("  • 位置在 (0, 0, 0)")
    print("  • Rotation & Scale 已 Apply")
    print("\n記得存檔！(Ctrl+S)")

# ============================================================
# 執行
# ============================================================

if __name__ == "__main__":
    fix_object_transforms()

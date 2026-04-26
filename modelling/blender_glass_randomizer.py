"""
Blender Glass Scene Randomizer
==============================

Generates randomized indoor scenes with transparent (glass) objects, designed
to be the input to the Mitsuba stereo renderer.

Features:
- Closed chamber (6 walls)
- Fixed table mode (supports multiple table sources)
- Floor furniture system (chairs, lamps, plants)
- Tabletop items (vases)
- Glass cup placement
- Random glass slabs (doors / windows)
- Texture system

Coordinate system:
- Origin: ground projection of the camera center
- X: left(-) <-> right(+)
- Y: depth   (0=camera, +=away)
- Z: height  (0=floor,  +=up)

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License.
"""

import bpy
import math
import random
import os
import re
from pathlib import Path
from mathutils import Vector
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================
# 配置參數
# ============================================================

CONFIG = {
    # 生成設定
    'num_scenes': 50,
    'output_dir': './scenes_output',  # 相對路徑，可用 --output 覆蓋
    'seed': None,
    'max_scene_attempts': 5,

    # 紋理設定
    'textures': {
        'enabled': True,
        'base_dir': './textures',  # relative to launch dir; override with --texture_dir
        'floor_textures': [
            'floor/laminate_floor_02_diff_8k.jpg',
            'floor/asphalt_06_diff_8k.jpg',
        ],
        'wall_texture': 'wall/plastered_wall_02_diff_8k.jpg',
        'randomize': {
            'floor': True,
            'wall_ceiling': True,
        },
        'preserve_original_materials': True,
    },

    # 場景尺寸 (mm)
    'chamber': {
        'width': 600.0,      # X: -300 ~ +300
        'height': 400.0,     # Z: 0 ~ 400
        'depth': 800.0,      # Y: 0 ~ 800
    },

    # 桌子設定 - 在渲染器視野內 (配合縮小 chamber)
    'table': {
        'enabled': True,
        'source_names': ['VintageTable2', 'VintageTable3'],  # VintageTable 沒有材質，不使用
        'position_x_range': (-30, 30),     # 縮小範圍
        'position_y_range': (300, 450),    # 調整深度範圍
        'rotation_z_range': (-30, 30),
        'shape': 'circle',
    },

    # 背景設定 (封閉 chamber)
    'background': {
        'enabled': True,
        'back_wall': {
            'enabled': True,
            'thickness': 5,
            'color': (0.75, 0.70, 0.65),
        },
        'front_wall': {
            'enabled': True,
            'y_position': 100,
            'thickness': 5,
            'color': (0.75, 0.70, 0.65),
        },
        'left_wall': {
            'enabled': True,
            'thickness': 5,
            'color': (0.72, 0.68, 0.63),
        },
        'right_wall': {
            'enabled': True,
            'thickness': 5,
            'color': (0.72, 0.68, 0.63),
        },
        'ground': {
            'enabled': True,
            'thickness': 2,
            'color': (0.45, 0.40, 0.35),
        },
        'ceiling': {
            'enabled': True,
            'thickness': 5,
            'color': (0.85, 0.83, 0.80),
        },
    },

    # 玻璃杯設定 (降低數量以減少 MC noise)
    'glass': {
        'count_range': (1, 1),  # 固定 1 個杯子
        'min_count': 1,
        'edge_margin': 8.0,
        'auto_discover': {
            'enabled': True,
            'collection': 'Copos',
            'sizes': {
                'TaçaChampagne': (6, 6, 22),
                'TaçaVinho': (10, 10, 23),
                'CopoShot': (4, 4, 5),
            },
            'default_size': (8, 8, 15),
            'blacklist': [],
        },
        'rotation_z': (0, 360),
        'scale_variation': (0.95, 1.05),
        'material_name': 'Glass_Clear',
    },

    # 隨機玻璃片設定 (放在地面上) - 優先放置
    # 相機在 Y=680，FOV=65°
    'glass_panels': {
        'enabled': True,
        'count_range': (2, 3),  # 玻璃片優先，2~3 片
        'min_count': 2,
        'width_range': (80, 200),
        'height_range': (150, 350),
        'thickness': 2.0,
        'x_range': (-120, 120),      # 配合縮小 chamber (X: ±300)
        'y_range': (150, 400),       # 配合縮小 chamber，相機 Y=680
        'rotation_z_range': (-45, 45),
        'collision_margin': 30.0,
    },

    # 碰撞檢測
    'collision_margin': 15,

    # 地板傢俱設定 (小物件)
    'ground_furniture': {
        'enabled': True,
        'count_range': (2, 4),
        'items': [
            {'name': 'brown chair', 'dims': (58, 58, 47)},
            {'name': 'VintageArmChair', 'dims': (63, 63, 109)},
            {'name': 'VintageChair', 'dims': (44, 51, 100)},
            {'name': 'VintageChair2', 'dims': (47, 49, 90)},
            {'name': 'VintageStool', 'dims': (42, 42, 54)},
            {'name': 'VintageStool2', 'dims': (38, 36, 45)},
            {'name': 'Stool_flow', 'dims': (30, 30, 26)},
            {'name': 'retro_radiator', 'dims': (48, 21, 62)},
            {'name': 'Gubi_Mategot_Trolley_mat(1)', 'dims': (48, 46, 37)},
            {'name': 'secto_design_octo_lamp_190sm_obj', 'dims': (22, 22, 78)},
        ],
        'rotation_z': (0, 360),
        'x_range': (-120, 120),
        'y_range': (150, 400),
    },

    # 櫃子設定 (大物件，獨立範圍)
    # 直接定死中心點範圍，不管物件尺寸
    'cabinets': {
        'enabled': True,
        'count_range': (1, 1),  # 一個場景只放一個櫃子
        'items': [
            # 只保留小型櫃子（寬度 < 100mm）
            {'name': 'Office_cabinet_1door_V1', 'dims': (60, 66, 78)},
            {'name': 'Office_cabinet_1door_V2', 'dims': (54, 53, 88)},
            {'name': 'Office_cabinet_2door_V1', 'dims': (60, 66, 113)},
            {'name': 'Office_cabinet_2door_V2', 'dims': (60, 66, 113)},
            {'name': 'Office_cabinet_tall', 'dims': (59, 65, 249)},
            # 大型物件移除：Office_cabinet_3door(200), Lockers(220), Office_desk(293)
        ],
        # 完全禁止旋轉，避免AABB膨脹
        'rotation_z': (0, 0),
        # 直接定死中心點範圍 (mm)
        # X: -150 ~ +150 (最大物件66mm寬，旋轉後最大約93mm，加上93/2=47，150+47=197 < 300 OK)
        # Y: 200 ~ 500 (前牆100，後面相機680)
        'center_x_range': (-150, 150),
        'center_y_range': (200, 500),
    },

    # 桌面小物設定
    'table_items': {
        'enabled': True,
        'count_range': (0, 2),
        'items': [
            # chrome_vase_v2 移除 - leaf.001 材質問題
            {'name': 'copper_vase', 'dims': (11, 20, 55)},
            {'name': 'skitsch_zucca_obj', 'dims': (54, 53, 23)},
        ],
        'rotation_z': (0, 360),
        'edge_margin': 10.0,
    },
}


# ============================================================
# Surface 資料結構
# ============================================================

@dataclass
class Surface:
    """可放置物品的表面"""
    parent_obj: object
    parent_name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_top: float
    placed_items: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    shape: str = 'rectangle'
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0

    @property
    def available_area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        return self.y_max - self.y_min

    def can_fit(self, item_w: float, item_d: float) -> bool:
        return item_w <= self.width and item_d <= self.depth

    def is_inside(self, x: float, y: float, half_w: float, half_d: float) -> bool:
        if self.shape == 'circle':
            dist_from_center = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            item_radius = math.sqrt(half_w**2 + half_d**2)
            return (dist_from_center + item_radius) <= self.radius
        else:
            return (
                (x - half_w) >= self.x_min and
                (x + half_w) <= self.x_max and
                (y - half_d) >= self.y_min and
                (y + half_d) <= self.y_max
            )

    def check_collision(self, x: float, y: float, aabb_w: float, aabb_d: float, margin: float) -> bool:
        for (ox, oy), (ow, od) in self.placed_items:
            if (abs(x - ox) < (aabb_w + ow) / 2 + margin and
                abs(y - oy) < (aabb_d + od) / 2 + margin):
                return True
        return False

    def add_item(self, x: float, y: float, aabb_w: float, aabb_d: float):
        self.placed_items.append(((x, y), (aabb_w, aabb_d)))


# ============================================================
# 命令行參數
# ============================================================

def parse_args():
    import sys
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    args = {'seed': None, 'count': None, 'output': None, 'texture_dir': None, 'diagnose': False}
    i = 0
    while i < len(argv):
        if argv[i] == '--seed' and i + 1 < len(argv):
            args['seed'] = int(argv[i + 1])
            i += 2
        elif argv[i] == '--count' and i + 1 < len(argv):
            args['count'] = int(argv[i + 1])
            i += 2
        elif argv[i] == '--output' and i + 1 < len(argv):
            args['output'] = argv[i + 1]
            i += 2
        elif argv[i] == '--texture_dir' and i + 1 < len(argv):
            args['texture_dir'] = argv[i + 1]
            i += 2
        elif argv[i] == '--diagnose':
            args['diagnose'] = True
            i += 1
        else:
            i += 1
    return args


def apply_args_to_config():
    args = parse_args()
    if args['seed'] is not None:
        CONFIG['seed'] = args['seed']
    if args['count'] is not None:
        CONFIG['num_scenes'] = args['count']
    if args['output'] is not None:
        CONFIG['output_dir'] = args['output']
    if args['texture_dir'] is not None:
        CONFIG['textures']['base_dir'] = args['texture_dir']


# ============================================================
# 紋理管理
# ============================================================

_scene_floor_texture: Optional[str] = None
_scene_wall_texture: Optional[str] = None


def get_texture_path(rel_path: str) -> Optional[str]:
    base = Path(CONFIG['textures']['base_dir'])
    full = base / rel_path
    return str(full) if full.exists() else None


def select_scene_textures():
    global _scene_floor_texture, _scene_wall_texture

    if CONFIG['textures']['randomize'].get('floor', False):
        floors = CONFIG['textures'].get('floor_textures', [])
        if floors:
            _scene_floor_texture = get_texture_path(random.choice(floors))

    if CONFIG['textures']['randomize'].get('wall_ceiling', False):
        wall = CONFIG['textures'].get('wall_texture', '')
        if wall:
            _scene_wall_texture = get_texture_path(wall)


# ============================================================
# 工具函數
# ============================================================

def get_rotated_aabb(width: float, depth: float, angle_rad: float) -> Tuple[float, float]:
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    return width * cos_a + depth * sin_a, width * sin_a + depth * cos_a


def snap_to_ground(obj):
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(c.z for c in corners)
    obj.location.z -= min_z


def snap_to_height(obj, target_z: float):
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(c.z for c in corners)
    obj.location.z += (target_z - min_z)


def get_object_surface_bounds(obj, surface_height_ratio: float, edge_margin: float) -> Optional[Surface]:
    """取得物件表面的可放置區域 - 使用 dimensions 計算"""
    bpy.context.view_layer.update()

    # 使用 dimensions（更可靠）
    dims = obj.dimensions
    loc = obj.location

    # 計算旋轉後的 AABB
    rot_z = obj.rotation_euler.z
    cos_r = abs(math.cos(rot_z))
    sin_r = abs(math.sin(rot_z))
    rotated_w = dims.x * cos_r + dims.y * sin_r
    rotated_d = dims.x * sin_r + dims.y * cos_r

    # 物件位置為中心（snap_to_ground 後底部在 Z=0）
    half_w = rotated_w / 2
    half_d = rotated_d / 2

    obj_x_min = loc.x - half_w
    obj_x_max = loc.x + half_w
    obj_y_min = loc.y - half_d
    obj_y_max = loc.y + half_d

    # 表面高度 = 物件高度（底部在 Z=0）
    z_top = dims.z * surface_height_ratio

    # 可放置區域
    x_min = obj_x_min + edge_margin
    x_max = obj_x_max - edge_margin
    y_min = obj_y_min + edge_margin
    y_max = obj_y_max - edge_margin

    print(f"    [Debug] dims={dims.x:.1f}x{dims.y:.1f}x{dims.z:.1f}, loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")
    print(f"    [Debug] surface: X[{x_min:.1f},{x_max:.1f}] Y[{y_min:.1f},{y_max:.1f}] Z={z_top:.1f}")

    if x_max <= x_min or y_max <= y_min:
        print(f"    [Debug] 表面太小: {x_max-x_min:.1f} x {y_max-y_min:.1f}")
        return None

    return Surface(
        parent_obj=obj,
        parent_name=obj.name,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_top=z_top,
    )


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_next_scene_index(output_dir):
    ensure_dir(output_dir)
    pattern = re.compile(r'scene_(\d+)')
    max_idx = 0
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


# ============================================================
# 材質建立
# ============================================================

def create_glass_material():
    name = CONFIG['glass']['material_name']
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["IOR"].default_value = 1.5
        mat.blend_method = 'BLEND'
    return mat


def create_diffuse_material(name: str, color: Tuple[float, float, float]):
    mat_name = f"Diffuse_{name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    return mat


def create_textured_material(name: str, texture_path: str, scale=(1.0, 1.0, 1.0)):
    mat_name = f"Textured_{name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    if not os.path.exists(texture_path):
        return None

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    tex_coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = scale
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.extension = 'REPEAT'

    try:
        img = bpy.data.images.load(texture_path, check_existing=True)
        tex_node.image = img
    except:
        return None

    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs["Roughness"].default_value = 0.8

    return mat


def get_material(name: str, color: Tuple[float, float, float], texture_path: Optional[str] = None):
    if texture_path and os.path.exists(texture_path):
        mat = create_textured_material(name, texture_path)
        if mat:
            return mat
    return create_diffuse_material(name, color)


# ============================================================
# 材質烘焙 (為沒有 Image Texture 的材質生成 texture)
# ============================================================

def fix_vintage_furniture_materials():
    """為 Vintage 系列家具綁定正確的 texture"""
    print("  [修復] 檢查 Vintage 家具材質...")

    texture_base = Path(CONFIG['textures']['base_dir']) / 'vintage_furniture'

    # 物件名稱 -> 資料夾名稱的映射
    vintage_mapping = {
        'VintageArmChair': 'VintageArmChair',
        'VintageChair': 'VintageChair',
        'VintageChair2': 'VintageChair2',
        'VintageStool': 'VintageStool',
        'VintageStool2': 'VintageStool',  # 用同一個
        'VintageTable': 'VintageTable',
        'VintageTable2': 'VintageTable2',
        'VintageTable3': 'VintageTable2',  # 用同一個
        'BarStool': 'BarStool',
        'BarTable': 'BarTable',
    }

    # 材質名稱 -> texture 檔案前綴的映射
    material_to_prefix = {
        'ArmChair': 'ArmChair',
        'Chair': 'Chair',
        'Chair.001': 'Chair',
        'Stool': 'Stool',
        'Stool.001': 'Stool',
        'Table': 'Table',
        'Table.001': 'Table',
    }

    fixed_count = 0

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        # 檢查是否是 Vintage 家具
        folder_name = None
        for prefix, folder in vintage_mapping.items():
            if obj.name.startswith(prefix) or obj.name == prefix:
                folder_name = folder
                break

        if not folder_name:
            continue

        texture_folder = texture_base / folder_name
        if not texture_folder.exists():
            print(f"    [!] 找不到資料夾: {texture_folder}")
            continue

        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue

            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if not bsdf:
                continue

            base_color = bsdf.inputs.get("Base Color")
            if not base_color:
                continue

            # 已經有 texture 連接，跳過
            if base_color.is_linked:
                continue

            # 找對應的 texture 檔案
            prefix = material_to_prefix.get(mat.name)
            if not prefix:
                # 嘗試從材質名稱推斷
                prefix = mat.name.replace('.001', '').replace('.002', '')

            # 尋找 Color texture
            color_tex = None
            for ext in ['.tga', '.png', '.jpg', '.jpeg']:
                candidate = texture_folder / f"{prefix}_Color{ext}"
                if candidate.exists():
                    color_tex = str(candidate)
                    break

            if not color_tex:
                # 嘗試其他命名方式
                for f in texture_folder.iterdir():
                    if 'color' in f.name.lower() or 'diffuse' in f.name.lower():
                        color_tex = str(f)
                        break

            if color_tex:
                # 創建 Image Texture 節點
                try:
                    img = bpy.data.images.load(color_tex, check_existing=True)
                    tex_node = nodes.new('ShaderNodeTexImage')
                    tex_node.image = img
                    tex_node.name = "ColorTexture"

                    # 連接到 Base Color
                    mat.node_tree.links.new(tex_node.outputs['Color'], base_color)

                    print(f"    [修復] {obj.name}/{mat.name} <- {Path(color_tex).name}")
                    fixed_count += 1
                except Exception as e:
                    print(f"    [!] 載入失敗 {color_tex}: {e}")

    print(f"  [修復] 完成，修復 {fixed_count} 個材質")


def fix_indirect_texture_connections():
    """修復有中間節點的 texture 連接 (e.g., Image -> Gamma -> Base Color)"""
    print("  [修復] 檢查間接 texture 連接...")
    fixed_count = 0

    for mat in bpy.data.materials:
        if not mat or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if not bsdf:
            continue

        base_color = bsdf.inputs.get("Base Color")
        if not base_color or not base_color.is_linked:
            continue

        # 找到連接到 Base Color 的節點
        source_node = None
        for link in links:
            if link.to_socket == base_color:
                source_node = link.from_node
                break

        if not source_node:
            continue

        # 如果已經是 Image Texture，跳過
        if source_node.type == 'TEX_IMAGE':
            continue

        # 追蹤找到 Image Texture (最多追 3 層)
        img_tex_node = None
        current_node = source_node
        for _ in range(3):
            # 找連接到這個節點的 Color/Image input
            found_input = None
            for inp in current_node.inputs:
                if inp.is_linked and inp.name in ['Color', 'Image', 'A', 'Color1', 'Fac']:
                    for link in links:
                        if link.to_socket == inp:
                            found_input = link.from_node
                            break
                    if found_input:
                        break

            if not found_input:
                break

            if found_input.type == 'TEX_IMAGE' and found_input.image:
                img_tex_node = found_input
                break

            current_node = found_input

        if img_tex_node:
            # 移除原有連接
            for link in list(links):
                if link.to_socket == base_color:
                    links.remove(link)
                    break

            # 直接連接 Image Texture -> Base Color
            links.new(img_tex_node.outputs['Color'], base_color)
            print(f"    [修復] {mat.name}: {img_tex_node.image.name} -> Base Color (跳過中間節點)")
            fixed_count += 1

    print(f"  [修復] 完成，修復 {fixed_count} 個間接連接")


def fix_indirect_texture_connections_silent():
    """靜默版本 - 匯出前修復，創建新的 Image Texture 節點直連 Base Color"""
    for mat in bpy.data.materials:
        if not mat or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if not bsdf:
            continue

        base_color = bsdf.inputs.get("Base Color")
        if not base_color or not base_color.is_linked:
            continue

        # 找到連接到 Base Color 的節點
        source_link = None
        for link in links:
            if link.to_socket == base_color:
                source_link = link
                break

        if not source_link:
            continue

        source_node = source_link.from_node

        # 如果已經是 Image Texture，跳過
        if source_node.type == 'TEX_IMAGE':
            continue

        # 追蹤找到 Image Texture
        img_tex_node = None
        current_node = source_node
        for _ in range(3):
            found_input = None
            for inp in current_node.inputs:
                if inp.is_linked and inp.name in ['Color', 'Image', 'A', 'Color1', 'Fac']:
                    for link in links:
                        if link.to_socket == inp:
                            found_input = link.from_node
                            break
                    if found_input:
                        break

            if not found_input:
                break

            if found_input.type == 'TEX_IMAGE' and found_input.image:
                img_tex_node = found_input
                break

            current_node = found_input

        if img_tex_node and img_tex_node.image:
            # 創建全新的 Image Texture 節點
            new_tex = nodes.new('ShaderNodeTexImage')
            new_tex.image = img_tex_node.image
            new_tex.name = "DirectColorTexture"
            new_tex.location = (bsdf.location.x - 300, bsdf.location.y)

            # 刪除原有連接
            links.remove(source_link)

            # 連接新節點到 Base Color
            links.new(new_tex.outputs['Color'], base_color)


def diagnose_materials():
    """診斷所有物件的材質狀態"""
    print("\n" + "=" * 60)
    print("[材質診斷]")
    print("=" * 60)

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if not obj.material_slots:
            continue

        print(f"\n物件: {obj.name}")
        for i, slot in enumerate(obj.material_slots):
            mat = slot.material
            if not mat:
                print(f"  [{i}] (空)")
                continue

            print(f"  [{i}] {mat.name}")

            if not mat.use_nodes:
                print(f"      - 不使用節點")
                continue

            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")

            if not bsdf:
                print(f"      - 無 Principled BSDF")
                continue

            base_color = bsdf.inputs.get("Base Color")
            if not base_color:
                print(f"      - 無 Base Color input")
                continue

            # 檢查是否有連接
            if base_color.is_linked:
                # 找到連接的節點
                for link in mat.node_tree.links:
                    if link.to_socket == base_color:
                        from_node = link.from_node
                        print(f"      - Base Color 連接自: {from_node.type} ({from_node.name})")

                        if from_node.type == 'TEX_IMAGE':
                            if from_node.image:
                                img = from_node.image
                                filepath = img.filepath if img.filepath else "(packed/generated)"
                                print(f"        Image: {img.name}")
                                print(f"        Path: {filepath}")
                                if img.filepath and not os.path.exists(bpy.path.abspath(img.filepath)):
                                    print(f"        ⚠️  檔案不存在!")
                            else:
                                print(f"        ⚠️  Image Texture 無圖片!")
                        break
            else:
                # 純色
                color = base_color.default_value
                print(f"      - Base Color 純色: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
                if color[0] == color[1] == color[2] == 0.5:
                    print(f"        ⚠️  默認灰色 (可能未設定)")
                elif color[0] == color[1] == color[2] == 0.8:
                    print(f"        ⚠️  Blender 默認值")

    print("\n" + "=" * 60)


# ============================================================
# 背景生成
# ============================================================

def create_backgrounds():
    chamber = CONFIG['chamber']
    bg = CONFIG['background']

    # 轉換 mm -> m
    width = chamber['width'] / 1000
    height = chamber['height'] / 1000
    depth = chamber['depth'] / 1000

    front_y = bg['front_wall'].get('y_position', 50) / 1000
    back_y = depth

    objects = []

    # 後牆
    if bg['back_wall']['enabled']:
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = "background_wall_back"
        thickness = bg['back_wall']['thickness'] / 1000
        obj.scale = (width, thickness, height)
        obj.location = (0, back_y, height / 2)
        bpy.ops.object.transform_apply(scale=True)
        mat = get_material('wall_back', bg['back_wall']['color'], _scene_wall_texture)
        obj.data.materials.append(mat)
        objects.append(obj)
        print(f"  + {obj.name}")

    # 前牆
    if bg['front_wall']['enabled']:
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = "background_wall_front"
        thickness = bg['front_wall']['thickness'] / 1000
        obj.scale = (width, thickness, height)
        obj.location = (0, front_y, height / 2)
        bpy.ops.object.transform_apply(scale=True)
        mat = get_material('wall_front', bg['front_wall']['color'], _scene_wall_texture)
        obj.data.materials.append(mat)
        objects.append(obj)
        print(f"  + {obj.name}")

    # 左牆
    if bg['left_wall']['enabled']:
        wall_depth = back_y - front_y
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = "background_wall_left"
        thickness = bg['left_wall']['thickness'] / 1000
        obj.scale = (thickness, wall_depth, height)
        obj.location = (-width / 2, front_y + wall_depth / 2, height / 2)
        bpy.ops.object.transform_apply(scale=True)
        mat = get_material('wall_left', bg['left_wall']['color'], _scene_wall_texture)
        obj.data.materials.append(mat)
        objects.append(obj)
        print(f"  + {obj.name}")

    # 右牆
    if bg['right_wall']['enabled']:
        wall_depth = back_y - front_y
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = "background_wall_right"
        thickness = bg['right_wall']['thickness'] / 1000
        obj.scale = (thickness, wall_depth, height)
        obj.location = (width / 2, front_y + wall_depth / 2, height / 2)
        bpy.ops.object.transform_apply(scale=True)
        mat = get_material('wall_right', bg['right_wall']['color'], _scene_wall_texture)
        obj.data.materials.append(mat)
        objects.append(obj)
        print(f"  + {obj.name}")

    # 地板
    if bg['ground']['enabled']:
        floor_depth = back_y - front_y
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = "background_ground"
        thickness = bg['ground']['thickness'] / 1000
        obj.scale = (width, floor_depth, thickness)
        obj.location = (0, front_y + floor_depth / 2, -thickness / 2)
        bpy.ops.object.transform_apply(scale=True)
        mat = get_material('ground', bg['ground']['color'], _scene_floor_texture)
        obj.data.materials.append(mat)
        objects.append(obj)
        print(f"  + {obj.name}")

    # 天花板
    if bg['ceiling']['enabled']:
        ceil_depth = back_y - front_y
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.name = "background_ceiling"
        thickness = bg['ceiling']['thickness'] / 1000
        obj.scale = (width, ceil_depth, thickness)
        obj.location = (0, front_y + ceil_depth / 2, height + thickness / 2)
        bpy.ops.object.transform_apply(scale=True)
        mat = get_material('ceiling', bg['ceiling']['color'], _scene_wall_texture)
        obj.data.materials.append(mat)
        objects.append(obj)
        print(f"  + {obj.name}")

    return objects


# ============================================================
# 桌子生成
# ============================================================

def create_table() -> Tuple[Optional[object], Optional[Surface]]:
    """生成桌子並取得表面"""
    cfg = CONFIG['table']
    if not cfg.get('enabled', False):
        return None, None

    # 隨機選擇桌子來源
    source_names = cfg.get('source_names', [])
    available_tables = [(name, bpy.data.objects.get(name)) for name in source_names]
    available_tables = [(n, o) for n, o in available_tables if o is not None]

    if not available_tables:
        print(f"  [!] 找不到桌子: {source_names}")
        return None, None

    source_name, source_obj = random.choice(available_tables)

    # 保存 source 的尺寸（複製後會丟失）
    source_dims = source_obj.dimensions.copy()
    print(f"  [Table] 使用: {source_name}")
    print(f"    原始尺寸: {source_dims.x*1000:.0f}x{source_dims.y*1000:.0f}x{source_dims.z*1000:.0f}mm")

    # 隨機位置和旋轉 (CONFIG 用 mm，轉換為 m)
    pos_x_mm = random.uniform(*cfg.get('position_x_range', (0, 0)))
    pos_y_mm = random.uniform(*cfg.get('position_y_range', (500, 500)))
    rot_z = math.radians(random.uniform(*cfg.get('rotation_z_range', (0, 0))))

    # 轉換為米
    pos_x = pos_x_mm / 1000
    pos_y = pos_y_mm / 1000

    # 複製物件
    obj = source_obj.copy()
    if source_obj.data:
        obj.data = source_obj.data.copy()
    bpy.context.collection.objects.link(obj)

    obj.name = "table_main"
    obj.hide_render = False
    obj.hide_viewport = False

    # 設定位置和旋轉 (米)
    obj.location = (pos_x, pos_y, 0)
    obj.rotation_euler = (0, 0, rot_z)
    bpy.context.view_layer.update()

    # 用 bound_box 貼地
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(c.z for c in corners)
    obj.location.z -= min_z  # 讓底部在 z=0
    bpy.context.view_layer.update()

    # 重新計算 bound_box 取得桌面高度
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    z_top = max(c.z for c in corners)  # 桌面實際 z 座標

    # 計算表面範圍
    edge_margin = CONFIG['glass']['edge_margin'] / 1000  # mm -> m

    # 從 bound_box 取得 XY 範圍
    all_x = [c.x for c in corners]
    all_y = [c.y for c in corners]
    obj_x_min = min(all_x)
    obj_x_max = max(all_x)
    obj_y_min = min(all_y)
    obj_y_max = max(all_y)

    # 計算旋轉後的尺寸（用於顯示）
    rotated_w = obj_x_max - obj_x_min
    rotated_d = obj_y_max - obj_y_min

    x_min = obj_x_min + edge_margin
    x_max = obj_x_max - edge_margin
    y_min = obj_y_min + edge_margin
    y_max = obj_y_max - edge_margin

    print(f"    桌面高度 z_top: {z_top*1000:.1f}mm")
    print(f"    表面計算: rotated={rotated_w*1000:.0f}x{rotated_d*1000:.0f}mm, surface={((x_max-x_min)*1000):.0f}x{((y_max-y_min)*1000):.0f}mm")

    if x_max > x_min and y_max > y_min:
        surface = Surface(
            parent_obj=obj,
            parent_name=obj.name,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_top=z_top,
        )

        # 根據桌子實際尺寸比例判斷形狀（而非固定 CONFIG）
        aspect_ratio = rotated_w / rotated_d if rotated_d > 0 else 1.0
        is_circular = 0.8 < aspect_ratio < 1.25  # 接近正方形視為圓形

        if is_circular:
            surface.shape = 'circle'
            surface.center_x = (obj_x_min + obj_x_max) / 2
            surface.center_y = (obj_y_min + obj_y_max) / 2
            surface.radius = min(rotated_w, rotated_d) / 2 - edge_margin
            print(f"    表面 (圓形): 中心=({surface.center_x*1000:.0f},{surface.center_y*1000:.0f}), 半徑={surface.radius*1000:.0f}mm")
        else:
            surface.shape = 'rectangle'
            print(f"    表面 (矩形): X[{x_min*1000:.0f}, {x_max*1000:.0f}] Y[{y_min*1000:.0f}, {y_max*1000:.0f}]")
    else:
        surface = None
        print(f"    [!] 表面太小")

    print(f"  + {obj.name} @ ({obj.location.x:.1f}, {obj.location.y:.1f}, {obj.location.z:.1f})")

    return obj, surface


# ============================================================
# 地板傢俱
# ============================================================

def place_ground_furniture(ground_occupied: list) -> list:
    """在地板上放置傢俱"""
    cfg = CONFIG['ground_furniture']
    if not cfg.get('enabled', False):
        return []

    items = cfg.get('items', [])
    if not items:
        return []

    min_count, max_count = cfg.get('count_range', (2, 4))
    target = random.randint(min_count, max_count)
    x_range = cfg.get('x_range', (-120, 120))    # 配合縮小 chamber
    y_range = cfg.get('y_range', (150, 400))     # 配合縮小 chamber
    rot_range = cfg.get('rotation_z', (0, 360))

    placed = []
    random.shuffle(items)

    # 轉換範圍為 m
    x_range_m = (x_range[0] / 1000, x_range[1] / 1000)
    y_range_m = (y_range[0] / 1000, y_range[1] / 1000)
    collision_margin_m = CONFIG['collision_margin'] / 1000

    for item in items:
        if len(placed) >= target:
            break

        source_obj = bpy.data.objects.get(item['name'])
        if source_obj is None:
            continue

        dims = source_obj.dimensions

        # 嘗試找位置 (全部使用 m 單位)
        for _ in range(30):
            rot_z = math.radians(random.uniform(*rot_range))
            aabb_w, aabb_d = get_rotated_aabb(dims.x, dims.y, rot_z)

            # X 方向可以稍微超出（不用全物件出鏡）
            # Y 方向需要確保物件在相機視野內
            half_d = aabb_d / 2
            eff_y_min = y_range_m[0] + half_d
            eff_y_max = y_range_m[1] - half_d

            # 如果物件太大放不下，跳過
            if eff_y_max <= eff_y_min:
                continue

            x = random.uniform(*x_range_m)  # X 方向直接用原範圍
            y = random.uniform(eff_y_min, eff_y_max)

            # 檢查碰撞 (全部 m 單位)
            collides = False
            for (ox, oy), (ow, od) in ground_occupied:
                if (abs(x - ox) < (aabb_w + ow) / 2 + collision_margin_m and
                    abs(y - oy) < (aabb_d + od) / 2 + collision_margin_m):
                    collides = True
                    break

            if collides:
                continue

            # 複製物件
            obj = source_obj.copy()
            if source_obj.data:
                obj.data = source_obj.data.copy()
            bpy.context.collection.objects.link(obj)

            obj.name = f"ground_{item['name']}"
            obj.hide_render = False
            obj.hide_viewport = False

            obj.location = (x, y, 0)
            obj.rotation_euler = (0, 0, rot_z)
            snap_to_ground(obj)

            ground_occupied.append(((x, y), (aabb_w, aabb_d)))
            placed.append(obj)
            print(f"  + {obj.name} @ ({x*1000:.1f}, {y*1000:.1f})")
            break

    return placed


# ============================================================
# 櫃子放置 (簡化版 - 直接定死範圍)
# ============================================================

def place_cabinets(ground_occupied: list) -> list:
    """在地板上放置櫃子 - 直接定死中心點範圍，不做複雜計算"""
    cfg = CONFIG.get('cabinets', {})
    if not cfg.get('enabled', False):
        return []

    items = cfg.get('items', [])
    if not items:
        return []

    min_count, max_count = cfg.get('count_range', (1, 1))
    target = random.randint(min_count, max_count)

    # 直接從 CONFIG 讀取定死的中心點範圍 (mm -> m)
    x_range = cfg.get('center_x_range', (-150, 150))
    y_range = cfg.get('center_y_range', (200, 500))
    x_min_m = x_range[0] / 1000
    x_max_m = x_range[1] / 1000
    y_min_m = y_range[0] / 1000
    y_max_m = y_range[1] / 1000

    # 禁止旋轉
    rot_z = 0

    collision_margin_m = CONFIG['collision_margin'] / 1000

    placed = []

    for _ in range(target):
        item = random.choice(items)

        source_obj = bpy.data.objects.get(item['name'])
        if source_obj is None:
            print(f"    [跳過] {item['name']} 不存在")
            continue

        # 固定尺寸 (mm -> m)
        w_m = item['dims'][0] / 1000
        d_m = item['dims'][1] / 1000

        placed_this = False
        for _ in range(30):
            # 直接在定死的範圍內隨機
            x = random.uniform(x_min_m, x_max_m)
            y = random.uniform(y_min_m, y_max_m)

            # 簡單碰撞檢測
            collides = False
            for (ox, oy), (ow, od) in ground_occupied:
                if (abs(x - ox) < (w_m + ow) / 2 + collision_margin_m and
                    abs(y - oy) < (d_m + od) / 2 + collision_margin_m):
                    collides = True
                    break
            if collides:
                continue

            # 放置物件
            obj = source_obj.copy()
            if source_obj.data:
                obj.data = source_obj.data.copy()
            bpy.context.collection.objects.link(obj)

            obj.name = f"ground_{item['name']}"
            obj.hide_render = False
            obj.hide_viewport = False

            # 先把原點設到幾何中心
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

            # 現在設置位置（原點在中心，所以物件中心會在指定位置）
            obj.location = (x, y, 0)
            obj.rotation_euler = (0, 0, rot_z)
            snap_to_ground(obj)

            ground_occupied.append(((x, y), (w_m, d_m)))
            placed.append(obj)
            placed_this = True

            # 輸出位置，方便確認
            print(f"  + {obj.name} @ ({x*1000:.0f}, {y*1000:.0f}) [範圍: X={x_range}, Y={y_range}]")
            break

        if not placed_this:
            print(f"    [跳過] {item['name']} 找不到位置")

    return placed


# ============================================================
# 桌面小物
# ============================================================

def place_table_items(surface: Surface) -> list:
    """在桌面上放置小物"""
    cfg = CONFIG['table_items']
    if not cfg.get('enabled', False) or surface is None:
        return []

    items = cfg.get('items', [])
    if not items:
        return []

    min_count, max_count = cfg.get('count_range', (0, 2))
    target = random.randint(min_count, max_count)
    rot_range = cfg.get('rotation_z', (0, 360))
    edge_margin = cfg.get('edge_margin', 10.0) / 1000  # mm -> m

    placed = []

    for _ in range(target * 3):
        if len(placed) >= target:
            break

        item = random.choice(items)
        source_obj = bpy.data.objects.get(item['name'])
        if source_obj is None:
            continue

        dims = source_obj.dimensions
        rot_z = math.radians(random.uniform(*rot_range))
        aabb_w, aabb_d = get_rotated_aabb(dims.x, dims.y, rot_z)

        half_w, half_d = aabb_w / 2, aabb_d / 2

        x_min = surface.x_min + half_w + edge_margin
        x_max = surface.x_max - half_w - edge_margin
        y_min = surface.y_min + half_d + edge_margin
        y_max = surface.y_max - half_d - edge_margin

        if x_max <= x_min or y_max <= y_min:
            continue

        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        # 圓形桌面檢查（避免物品懸空）
        if surface.shape == 'circle':
            item_radius = math.sqrt(half_w**2 + half_d**2)
            dist = math.sqrt((x - surface.center_x)**2 + (y - surface.center_y)**2)
            if dist + item_radius > surface.radius - edge_margin:
                continue

        if surface.check_collision(x, y, aabb_w, aabb_d, CONFIG['collision_margin'] / 1000):
            continue

        # 複製物件
        obj = source_obj.copy()
        if source_obj.data:
            obj.data = source_obj.data.copy()
        bpy.context.collection.objects.link(obj)

        obj.name = f"table_item_{item['name']}"
        obj.hide_render = False
        obj.hide_viewport = False

        obj.location = (x, y, 0)
        obj.rotation_euler = (0, 0, rot_z)
        snap_to_height(obj, surface.z_top)

        surface.add_item(x, y, aabb_w, aabb_d)
        placed.append(obj)
        print(f"  + {obj.name} @ ({x:.1f}, {y:.1f}, {surface.z_top:.1f})")

    return placed


# ============================================================
# 玻璃杯
# ============================================================

def discover_glass_objects() -> list:
    """從 Collection 搜尋玻璃物件"""
    cfg = CONFIG['glass']['auto_discover']
    if not cfg.get('enabled', False):
        return []

    col_name = cfg.get('collection', 'Copos')
    col = bpy.data.collections.get(col_name)
    if col is None:
        return []

    sizes = cfg.get('sizes', {})
    default_size = cfg.get('default_size', (8, 8, 15))
    blacklist = cfg.get('blacklist', [])

    available = []
    for obj in col.objects:
        if obj.type != 'MESH':
            continue
        if obj.name in blacklist:
            continue

        size = sizes.get(obj.name, default_size)
        available.append({
            'name': obj.name,
            'obj_name': obj.name,
            'size': size,
        })
        print(f"  ✓ {obj.name} (玻璃, {size[0]}x{size[1]}x{size[2]}mm)")

    return available


def place_glass_on_surface(glass_list: list, surface: Surface) -> list:
    """在桌面上放置玻璃杯"""
    if not glass_list or surface is None:
        print("    [!] 無玻璃或無表面")
        return []

    cfg = CONFIG['glass']
    min_count, max_count = cfg.get('count_range', (2, 5))
    target = random.randint(min_count, max_count)
    rot_range = cfg.get('rotation_z', (0, 360))
    edge_margin = cfg.get('edge_margin', 8.0) / 1000  # mm -> m
    collision_margin = CONFIG['collision_margin'] / 1000

    # 取得桌面中心和範圍
    table_center_x = (surface.x_min + surface.x_max) / 2
    table_center_y = (surface.y_min + surface.y_max) / 2
    table_z = surface.z_top

    # 可用範圍（扣掉邊緣）
    usable_x_min = surface.x_min + edge_margin
    usable_x_max = surface.x_max - edge_margin
    usable_y_min = surface.y_min + edge_margin
    usable_y_max = surface.y_max - edge_margin

    print(f"    桌面中心: ({table_center_x*1000:.1f}, {table_center_y*1000:.1f}) mm")
    print(f"    可用範圍: X[{usable_x_min*1000:.1f}, {usable_x_max*1000:.1f}] Y[{usable_y_min*1000:.1f}, {usable_y_max*1000:.1f}] mm")

    if usable_x_max <= usable_x_min or usable_y_max <= usable_y_min:
        print("    [!] 桌面太小，無法放置玻璃杯")
        return []

    placed = []
    placed_positions = []  # (x, y, radius)

    for _ in range(target * 10):
        if len(placed) >= target:
            break

        glass_info = random.choice(glass_list)
        source_obj = bpy.data.objects.get(glass_info['obj_name'])
        if source_obj is None:
            continue

        # 玻璃杯尺寸 (mm -> m)
        glass_w = glass_info['size'][0] / 1000
        glass_d = glass_info['size'][1] / 1000
        glass_h = glass_info['size'][2] / 1000
        glass_radius = max(glass_w, glass_d) / 2

        # 隨機位置（在可用範圍內）
        x = random.uniform(usable_x_min + glass_radius, usable_x_max - glass_radius)
        y = random.uniform(usable_y_min + glass_radius, usable_y_max - glass_radius)

        # 圓形桌面檢查
        if surface.shape == 'circle':
            dist = math.sqrt((x - table_center_x)**2 + (y - table_center_y)**2)
            if dist + glass_radius > surface.radius - edge_margin:
                continue

        # 碰撞檢測
        collides = False
        for px, py, pr in placed_positions:
            if math.sqrt((x - px)**2 + (y - py)**2) < (glass_radius + pr + collision_margin):
                collides = True
                break

        if collides:
            continue

        # 複製物件
        obj = source_obj.copy()
        if source_obj.data:
            obj.data = source_obj.data.copy()
        bpy.context.collection.objects.link(obj)

        obj.name = f"glass_{glass_info['name']}_{len(placed):02d}"
        obj.hide_render = False
        obj.hide_viewport = False

        # 直接放置在桌面上（不做額外縮放，保持原尺寸）
        rot_z = math.radians(random.uniform(*rot_range))

        # 先設定旋轉（確保正面朝上）
        obj.rotation_euler = (math.pi, 0, rot_z)

        # 設定位置
        obj.location = (x, y, table_z)
        bpy.context.view_layer.update()

        # 調整 Z 位置讓杯底在桌面上
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        min_z = min(c.z for c in corners)
        obj.location.z += (table_z - min_z)

        # 套用玻璃材質
        mat = create_glass_material()
        if obj.data.materials:
            obj.data.materials.clear()
        obj.data.materials.append(mat)

        placed_positions.append((x, y, glass_radius))
        placed.append(obj)
        print(f"  + {obj.name} @ ({x*1000:.1f}, {y*1000:.1f}, {table_z*1000:.1f}) mm")

    return placed


# ============================================================
# 玻璃片
# ============================================================

def place_glass_panels(ground_occupied: list) -> list:
    """放置隨機玻璃片在地面上"""
    cfg = CONFIG['glass_panels']
    if not cfg.get('enabled', False):
        return []

    min_count, max_count = cfg.get('count_range', (1, 2))
    target = random.randint(min_count, max_count)

    # 轉換範圍為 m (配合縮小 chamber)
    x_range = cfg.get('x_range', (-120, 120))
    y_range = cfg.get('y_range', (150, 400))
    x_range_m = (x_range[0] / 1000, x_range[1] / 1000)
    y_range_m = (y_range[0] / 1000, y_range[1] / 1000)
    collision_margin_m = cfg.get('collision_margin', 25.0) / 1000

    placed = []

    for idx in range(target):
        width = random.uniform(*cfg.get('width_range', (60, 150))) / 1000
        height = random.uniform(*cfg.get('height_range', (100, 250))) / 1000
        thickness = cfg.get('thickness', 1.5) / 1000

        rot_z_range = cfg.get('rotation_z_range', (-45, 45))

        # 嘗試找位置
        for _ in range(50):
            x = random.uniform(*x_range_m)
            y = random.uniform(*y_range_m)
            rot_z = math.radians(random.uniform(*rot_z_range))

            # 計算旋轉後的 AABB (玻璃片在 XY 平面的佔用)
            aabb_w, aabb_d = get_rotated_aabb(width, thickness, rot_z)

            # 檢查與地板傢俱碰撞
            collides = False
            for (ox, oy), (ow, od) in ground_occupied:
                if (abs(x - ox) < (aabb_w + ow) / 2 + collision_margin_m and
                    abs(y - oy) < (aabb_d + od) / 2 + collision_margin_m):
                    collides = True
                    break

            if collides:
                continue

            # 建立玻璃片
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.active_object
            obj.name = f"glass_panel_{idx:02d}"
            obj.scale = (width, thickness, height)

            bpy.ops.object.transform_apply(scale=True)

            # 放在地面上 (底部 z=0)
            obj.location = (x, y, height / 2)
            obj.rotation_euler = (0, 0, rot_z)

            mat = create_glass_material()
            obj.data.materials.append(mat)

            # 加入碰撞列表
            ground_occupied.append(((x, y), (aabb_w, aabb_d)))
            placed.append(obj)
            print(f"  + {obj.name} ({width*1000:.0f}x{height*1000:.0f}mm) @ ({x*1000:.1f}, {y*1000:.1f})")
            break

    return placed


# ============================================================
# 清理和匯出
# ============================================================

def clear_generated():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith(('glass_', 'diffuse_', 'background_', 'ground_', 'table_')):
            if obj.type == 'MESH':
                obj.select_set(True)
    bpy.ops.object.delete()


def export_scene(scene_id, output_dir):
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f"scene_{scene_id:04d}.obj")

    # 匯出前再次修復材質連接
    fix_indirect_texture_connections_silent()

    # 強制更新所有材質
    for mat in bpy.data.materials:
        if mat and mat.use_nodes:
            mat.node_tree.update_tag()
    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith(('glass_', 'diffuse_', 'background_', 'ground_', 'table_')):
            if obj.type == 'MESH':
                obj.select_set(True)

    if not bpy.context.selected_objects:
        print(f"  [!] 沒有物件可匯出")
        return None

    try:
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            forward_axis='Y',
            up_axis='Z',
            global_scale=1000.0,  # m -> mm
            apply_modifiers=True,
            export_materials=True,
            path_mode='COPY',
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            axis_forward='Y',
            axis_up='Z',
            global_scale=1000.0,  # m -> mm
            use_materials=True,
            path_mode='COPY',
        )

    print(f"  -> 匯出: {filepath}")
    return filepath


# ============================================================
# 場景生成
# ============================================================

def generate_single_scene(available_glass: list) -> Tuple[bool, int]:
    clear_generated()
    select_scene_textures()

    ground_occupied = []
    glass_count = 0

    # 1. 背景
    print("  [背景]")
    create_backgrounds()

    # 2. 桌子
    print("  [桌子]")
    table_obj, table_surface = create_table()

    if table_obj:
        dims = table_obj.dimensions
        aabb_w, aabb_d = dims.x, dims.y
        center_x = table_obj.location.x
        center_y = table_obj.location.y
        ground_occupied.append(((center_x, center_y), (aabb_w, aabb_d)))

    # 3. 玻璃片 (優先放置，確保有空間)
    print("  [玻璃片]")
    panel_items = place_glass_panels(ground_occupied)
    glass_count += len(panel_items)
    print(f"    放置 {len(panel_items)} 個")

    # 4. 地板傢俱 (玻璃片之後放置)
    print("  [地板傢俱]")
    ground_items = place_ground_furniture(ground_occupied)
    print(f"    放置 {len(ground_items)} 個")

    # 5. 櫃子 (獨立範圍，可部分出鏡)
    print("  [櫃子]")
    cabinet_items = place_cabinets(ground_occupied)
    print(f"    放置 {len(cabinet_items)} 個")

    # 7. 桌面小物
    if table_surface:
        print("  [桌面小物]")
        table_items = place_table_items(table_surface)
        print(f"    放置 {len(table_items)} 個")

    # 8. 玻璃杯
    if table_surface and available_glass:
        print("  [玻璃杯]")
        glass_items = place_glass_on_surface(available_glass, table_surface)
        glass_count += len(glass_items)
        print(f"    放置 {len(glass_items)} 個")

    # 檢查玻璃數量
    min_glass = CONFIG['glass']['min_count']
    min_panels = CONFIG['glass_panels'].get('min_count', 0)
    total_min = min_glass + min_panels

    if glass_count < total_min:
        print(f"  [!] 玻璃數量不足: {glass_count}/{total_min}")
        return False, glass_count

    return True, glass_count


# ============================================================
# 主程式
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("Blender Glass Scene Randomizer")
    print("=" * 60)

    args = parse_args()

    # 診斷模式
    if args.get('diagnose', False):
        diagnose_materials()
        return

    apply_args_to_config()

    # 設定單位
    unit = bpy.context.scene.unit_settings
    unit.system = 'METRIC'
    unit.length_unit = 'MILLIMETERS'
    unit.scale_length = 0.001

    # 修復 Vintage 家具材質
    print("\n修復材質...")
    fix_vintage_furniture_materials()
    fix_indirect_texture_connections()

    # 搜尋玻璃物件
    print("\n檢查玻璃物件...")
    available_glass = discover_glass_objects()

    if not available_glass:
        print("[警告] 沒有可用的玻璃物件！")

    output_dir = CONFIG['output_dir']
    start_idx = get_next_scene_index(output_dir)
    num_scenes = CONFIG['num_scenes']

    print(f"\n輸出目錄: {output_dir}")
    print(f"場景範圍: {start_idx} ~ {start_idx + num_scenes - 1}")
    print(f"可用玻璃: {len(available_glass)} 種")
    print("=" * 60)

    success_count = 0
    total_glass = 0

    for i in range(num_scenes):
        scene_idx = start_idx + i
        seed = (CONFIG['seed'] or 100) + i

        print(f"\n[場景 {scene_idx}] seed={seed}")

        for attempt in range(CONFIG['max_scene_attempts']):
            random.seed(seed + attempt * 1000)

            success, glass_count = generate_single_scene(available_glass)

            if success:
                export_scene(scene_idx, output_dir)
                print(f"  ✓ 成功 (玻璃: {glass_count})")
                success_count += 1
                total_glass += glass_count
                break
            else:
                print(f"  重試 {attempt + 1}/{CONFIG['max_scene_attempts']}")
        else:
            print(f"  ✗ 失敗")

    print("\n" + "=" * 60)
    print(f"完成: {success_count}/{num_scenes} 場景")
    print(f"總玻璃數: {total_glass}")
    print("=" * 60)


if __name__ == "__main__":
    main()

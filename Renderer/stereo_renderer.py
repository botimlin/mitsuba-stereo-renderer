"""
Mitsuba Stereo Renderer (RGB + Camera-centric Coordinate System)
================================================================

A standalone GPU stereo renderer for indoor scenes containing
transparent (glass) objects. Built on Mitsuba 3 (`cuda_ad_rgb`).

Coordinate System
-----------------

Origin: ground projection of the camera center.
- Camera actual position: (0, 0, CAMERA_HEIGHT)

Scene axes:
- X: left(-) <-> right(+)        [horizontal]
- Y: depth   (0=camera, +=away)  [depth]
- Z: height  (0=floor,  +=up)    [height]

Scene layout:
                Y+ (depth)
                ^
                |   back wall @ Y=CHAMBER_DEPTH
                |   ================
                |
                |   table @ Y=TABLE_DEPTH
                |     +-----+
                |     |     |
       X- <-----+-----+-----+--->  X+
                |
                |   camera @ Y=CAMERA_DEPTH
                +-----------------------+
                |   (0, 680, 100)       |
                +-----------------------+

Default key parameters (mm):
- CAMERA_DEPTH = 680         (camera depth, near back wall)
- CAMERA_HEIGHT = 100        (camera height above floor)
- TABLE_DEPTH = 350          (camera-to-table depth, lookat target)
- TABLE_SURFACE_HEIGHT = 100 (lookat height; matches camera height)
- CHAMBER_DEPTH = 800        (max scene depth)
- CHAMBER_WIDTH = 600        (scene width)
- CHAMBER_HEIGHT = 400       (ceiling height)

Blender/OBJ -> Mitsuba transform:
- Mesh transform: scale(0.001) @ rotate([1,0,0], -90)
- Effect: Scene (X, Y, Z) -> Mitsuba (X, Z, -Y)
- Mitsuba axes: X(right), Y(up), Z(negative = forward)

Output format
-------------
{output_dir}/{scene_name}/
├── left.exr               # Left camera RGB  (float32)
├── right.exr              # Right camera RGB (float32)
├── depth.exr              # GT depth (left viewpoint, meters)
├── disparity.exr          # GT disparity (pixels)
├── glass_mask.png         # Glass region (intersection of left/right)
├── params.json            # Render parameters + seed + SPP
└── preview/
    ├── left.png
    └── right.png
"""

from __future__ import annotations

import numpy as np
import cv2
import os
import json
import argparse
import gc
import math
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib
import sys

# Lazy import mitsuba to avoid CUDA init in main process
mi = None
dr = None

def lazy_import_mitsuba():
    """Lazy load mitsuba, only initialize when needed"""
    global mi, dr
    if mi is None:
        import mitsuba as _mi
        import drjit as _dr
        mi = _mi
        dr = _dr


def deterministic_hash(s: str) -> int:
    """Deterministic hash, consistent across Python sessions"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


# ============================================================
# Configuration
# ============================================================

class Config:
    """
    Render configuration - all units in mm unless noted

    Coordinate system:
    - Origin: ground projection of the camera center
    - X: left/right (right is positive)
    - Y: depth (away from camera is positive)
    - Z: height (up is positive)
    """

    # ============================================================
    # Render
    # ============================================================
    WIDTH = 640
    HEIGHT = 480
    SPP = 1024            # Samples per pixel
    SPP_PER_BATCH = 64    # GPU memory batching
    MAX_DEPTH = 16        # Ray bounce count (reduced for faster convergence)

    # Denoiser 設定 (OptiX AI Denoiser)
    DENOISE = False       # 關閉 denoiser，靠 SPP 收斂

    # ============================================================
    # Scene size (Scene Coordinates, mm)
    # ============================================================
    CHAMBER_WIDTH = 600.0
    CHAMBER_HEIGHT = 400.0
    CHAMBER_DEPTH = 800.0

    # ============================================================
    # Camera (Scene Coordinates)
    # ============================================================
    # Camera sits near the back wall and looks toward the front of the scene.
    # The LED panel is mounted on the back wall (behind the camera).
    CAMERA_DEPTH = 680.0
    CAMERA_HEIGHT = 100.0
    BASELINE = 65.0           # stereo baseline (mm)

    # Per-scene camera jitter
    CAMERA_RANDOMIZE = True
    CAMERA_X_JITTER = 15.0
    CAMERA_DEPTH_JITTER = 20.0    # back-only jitter (positive offset)
    CAMERA_HEIGHT_JITTER = 10.0
    LOOKAT_X_JITTER = 5.0
    LOOKAT_Z_JITTER = 5.0

    # Gentle glass-tracking lookat: rotate baseline center toward glass centroid.
    GLASS_LOOKAT_ENABLED = True
    GLASS_LOOKAT_MAX_X = 30.0     # cap on horizontal offset (mm)
    GLASS_LOOKAT_STRENGTH = 0.3   # 0..1 (fraction of glass centroid X)

    # Runtime camera state (per-scene)
    _camera_x_offset = 0.0
    _camera_depth_offset = 0.0
    _camera_height_offset = 0.0
    _lookat_x_offset = 0.0
    _lookat_z_offset = 0.0
    _glass_lookat_x_offset = 0.0

    # ============================================================
    # Sensor (Sony IMX296LQR-C)
    # ============================================================
    SENSOR_WIDTH = 5.023      # mm
    SENSOR_HEIGHT = 3.754     # mm
    FOCAL_LENGTH = 6.0        # mm
    FOV = 45.4                # degrees

    # ============================================================
    # Lookat target / table (Scene Coordinates)
    # ============================================================
    TABLE_DEPTH = 350.0
    TABLE_SURFACE_HEIGHT = 100.0

    TABLE_DEPTH_RANGE = (300.0, 500.0)
    TABLE_X_RANGE = (-100.0, 100.0)

    # Glass depth range (relative to camera)
    GLASS_DEPTH_MIN = 300.0
    GLASS_DEPTH_MAX = 900.0

    # ============================================================
    # Lighting (back-wall LED panel)
    # ============================================================
    LED_INTENSITY = 25000000.0

    # LED covers the back wall.
    LED_SIZE = (600.0, 400.0)

    LED_POSITION = (0.0, 750.0, 200.0)    # back wall, inside the chamber
    LED_TARGET = (0.0, 300.0, 200.0)      # aim at scene center

    # Ambient
    AMBIENT_INTENSITY = 0.0

    # Ceiling fill lights.
    CEILING_LIGHTS_ENABLED = True
    CEILING_EMITTER_INTENSITY = 10000000.0
    CEILING_LIGHT_POSITIONS = [
        (-120.0, 300.0, 380.0),   # 前左 (配合縮小 chamber)
        (120.0, 300.0, 380.0),    # 前右
        (-120.0, 500.0, 380.0),   # 後左
        (120.0, 500.0, 380.0),    # 後右
    ]

    # ============================================================
    # 材質設定
    # ============================================================
    GLASS_IOR = 1.5
    GLASS_ROUGHNESS = 0.02
    GLASS_TYPE = 'thick'  # 'thin' or 'thick'

    # ============================================================
    # 輸出設定
    # ============================================================
    SAVE_PREVIEW = True
    DOLP_FLOOR = 0.001  # QA threshold

    # ============================================================
    # 相機位置計算方法
    # ============================================================

    @classmethod
    def randomize_camera(cls, rng: Optional[np.random.Generator] = None):
        """
        為此場景採樣隨機相機偏移。
        每個場景渲染前呼叫一次。
        """
        if not cls.CAMERA_RANDOMIZE:
            cls._camera_depth_offset = 0.0
            cls._camera_height_offset = 0.0
            cls._lookat_x_offset = 0.0
            cls._lookat_z_offset = 0.0
            return

        if rng is None:
            rng = np.random.default_rng()

        cls._camera_x_offset = rng.uniform(-cls.CAMERA_X_JITTER, cls.CAMERA_X_JITTER)  # 左右
        cls._camera_depth_offset = rng.uniform(0, cls.CAMERA_DEPTH_JITTER)  # 只能後移 (正值)
        cls._camera_height_offset = rng.uniform(-cls.CAMERA_HEIGHT_JITTER, cls.CAMERA_HEIGHT_JITTER)
        cls._lookat_x_offset = rng.uniform(-cls.LOOKAT_X_JITTER, cls.LOOKAT_X_JITTER)
        cls._lookat_z_offset = rng.uniform(-cls.LOOKAT_Z_JITTER, cls.LOOKAT_Z_JITTER)
        cls._glass_lookat_x_offset = 0.0  # 重置，等待 apply_glass_lookat 設定

    @classmethod
    def apply_glass_lookat(cls, glass_centroid_x: float):
        """
        根據玻璃物體質心位置，微微調整相機朝向。

        這是非常克制的調整：
        - 只調整水平方向 (X)
        - 以基線中心為旋轉中心
        - 最大偏移受 GLASS_LOOKAT_MAX_X 限制
        - 偏移強度受 GLASS_LOOKAT_STRENGTH 控制

        Args:
            glass_centroid_x: 玻璃物體質心的 X 座標 (mm, Scene coords)
        """
        if not cls.GLASS_LOOKAT_ENABLED:
            cls._glass_lookat_x_offset = 0.0
            return

        # 計算需要的偏移量（往玻璃方向移動）
        raw_offset = glass_centroid_x * cls.GLASS_LOOKAT_STRENGTH

        # 限制最大偏移
        clamped_offset = np.clip(raw_offset, -cls.GLASS_LOOKAT_MAX_X, cls.GLASS_LOOKAT_MAX_X)

        cls._glass_lookat_x_offset = clamped_offset

    @classmethod
    def get_camera_offsets(cls) -> Dict[str, float]:
        """取得目前相機偏移量（用於 logging）"""
        return {
            'camera_x_offset': cls._camera_x_offset,
            'camera_depth_offset': cls._camera_depth_offset,
            'camera_height_offset': cls._camera_height_offset,
            'lookat_x_offset': cls._lookat_x_offset,
            'lookat_z_offset': cls._lookat_z_offset,
            'glass_lookat_x_offset': cls._glass_lookat_x_offset,
        }

    @classmethod
    def camera_depth(cls) -> float:
        """相機 Y 位置（含隨機化）- 在前牆後面"""
        return cls.CAMERA_DEPTH + cls._camera_depth_offset

    @classmethod
    def camera_height(cls) -> float:
        """相機 Z 位置（含隨機化）"""
        return cls.CAMERA_HEIGHT + cls._camera_height_offset

    @classmethod
    def camera_x(cls) -> float:
        """相機 X 位置（含隨機化）"""
        return cls._camera_x_offset

    @classmethod
    def camera_position(cls) -> Tuple[float, float, float]:
        """相機中心位置 (Scene Coordinates)"""
        return (cls.camera_x(), cls.camera_depth(), cls.camera_height())

    @classmethod
    def left_camera_position(cls) -> Tuple[float, float, float]:
        """左相機位置 (Scene Coordinates)"""
        return (cls.camera_x() - cls.BASELINE / 2, cls.camera_depth(), cls.camera_height())

    @classmethod
    def right_camera_position(cls) -> Tuple[float, float, float]:
        """右相機位置 (Scene Coordinates)"""
        return (cls.camera_x() + cls.BASELINE / 2, cls.camera_depth(), cls.camera_height())

    @classmethod
    def target_point(cls) -> Tuple[float, float, float]:
        """相機看向的點（桌面中心，含隨機化 + 玻璃追蹤）"""
        # 總 X 偏移 = 隨機抖動 + 玻璃追蹤
        total_lookat_x = cls._lookat_x_offset + cls._glass_lookat_x_offset
        return (
            0.0 + total_lookat_x,
            cls.TABLE_DEPTH,
            cls.TABLE_SURFACE_HEIGHT + cls._lookat_z_offset
        )

    @classmethod
    def forward_direction(cls) -> Tuple[float, float, float]:
        """相機光軸方向（歸一化）"""
        cam_pos = np.array(cls.camera_position())
        target = np.array(cls.target_point())
        direction = target - cam_pos
        direction = direction / np.linalg.norm(direction)
        return tuple(direction)

    @classmethod
    def depth_camera_position(cls) -> Tuple[float, float, float]:
        """深度相機位置（與左相機相同）"""
        return cls.left_camera_position()

    @classmethod
    def camera_target_for_position(cls, camera_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """計算各相機的目標點（確保平行光軸）"""
        direction = np.array(cls.forward_direction())
        cam_center = np.array(cls.camera_position())
        target = np.array(cls.target_point())
        distance = np.linalg.norm(target - cam_center)
        camera_pos_arr = np.array(camera_pos)
        camera_target = camera_pos_arr + direction * distance
        return tuple(camera_target)


def mm_to_m(mm: float) -> float:
    """Millimeters to meters"""
    return mm / 1000.0


# ============================================================
# Mitsuba Setup (cuda_ad_rgb)
# ============================================================

def setup_mitsuba() -> str:
    """Set up Mitsuba variant (RGB)."""
    lazy_import_mitsuba()
    available = mi.variants()
    print(f"[Mitsuba] Available variants: {available}")

    preferred = [
        'cuda_ad_rgb',
        'cuda_rgb',
        'llvm_ad_rgb',
        'llvm_rgb',
        'scalar_rgb',
    ]

    for variant in preferred:
        if variant in available:
            try:
                if 'cuda' in variant:
                    dr.set_flag(dr.JitFlag.Debug, False)
                mi.set_variant(variant)
                print(f"[Mitsuba] Using variant: {variant}")
                return variant
            except ImportError as e:
                print(f"[Mitsuba] {variant} init failed: {e}")
                continue

    raise RuntimeError(
        f"Cannot find a usable RGB variant!\n"
        f"Available: {available}"
    )


# ============================================================
# RGB Helpers
# ============================================================

def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) -> luminance (H,W), ITU-R BT.709 weights."""
    if rgb.ndim == 2:
        return rgb
    return (0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]).astype(np.float32)


# ============================================================
# Material Factory
# ============================================================

class MaterialFactory:
    """Material creation factory"""

    @staticmethod
    def twosided(bsdf: Dict) -> Dict:
        """Wrap BSDF with twosided for interior rendering"""
        return {
            'type': 'twosided',
            'bsdf': bsdf,
        }

    @staticmethod
    def glass(ior: float = Config.GLASS_IOR, thin: bool = None) -> Dict:
        """Create glass material"""
        if thin is None:
            use_thin = (Config.GLASS_TYPE == 'thin')
        else:
            use_thin = thin

        if use_thin:
            return {
                'type': 'thindielectric',
                'int_ior': ior,
            }
        else:
            return {
                'type': 'dielectric',
                'int_ior': ior,
            }

    @staticmethod
    def diffuse(reflectance: float) -> Dict:
        """Create diffuse material"""
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'spectrum',
                'value': reflectance,
            },
        }

    @staticmethod
    def diffuse_rgb(color: Tuple[float, float, float]) -> Dict:
        """Create diffuse material from RGB color"""
        # 使用 srgb spectrum 保留顏色信息
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'srgb',
                'color': list(color),
            },
        }

    @staticmethod
    def diffuse_textured(texture_path: str) -> Dict:
        """Create textured diffuse material"""
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': texture_path,
                'filter_type': 'bilinear',
                'wrap_mode': 'repeat',
            },
        }


# ============================================================
# MTL Parser (supports textures)
# ============================================================

class MTLParser:
    """OBJ MTL material file parser"""

    NON_GLASS_KEYWORDS = [
        'background', 'wall', 'floor', 'ground', 'ceiling',
        'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric',
        'concrete', 'brick', 'stone', 'plastic', 'rubber',
    ]

    GLASS_EXACT_NAMES = ['glass_clear']

    @classmethod
    def parse(cls, mtl_path: str) -> Dict[str, Dict]:
        """Parse MTL file"""
        materials = {}
        current = None
        mtl_dir = os.path.dirname(os.path.abspath(mtl_path))

        if not os.path.exists(mtl_path):
            print(f"  [MTL] Not found: {mtl_path}")
            return materials

        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0].lower()

                if cmd == 'newmtl' and len(parts) > 1:
                    current = parts[1]
                    materials[current] = {
                        'is_glass': cls._is_glass_name(current),
                        'color': (0.5, 0.5, 0.5),
                        'textures': {'diffuse': None},
                    }
                elif current:
                    if cmd == 'kd' and len(parts) >= 4:
                        materials[current]['color'] = (
                            float(parts[1]), float(parts[2]), float(parts[3]),
                        )
                    elif cmd == 'd' and len(parts) >= 2:
                        if float(parts[1]) < 0.95:
                            materials[current]['is_glass'] = True
                    elif cmd == 'illum' and len(parts) >= 2:
                        if int(parts[1]) in [4, 6, 7, 9]:
                            materials[current]['is_glass'] = True
                    elif cmd == 'map_kd' and len(parts) >= 2:
                        tex_path_raw = ' '.join(parts[1:])
                        tex_abs_path = cls._resolve_texture_path(tex_path_raw, mtl_dir)
                        if tex_abs_path:
                            materials[current]['textures']['diffuse'] = tex_abs_path
                        else:
                            print(f"  [MTL] WARNING: texture not found: {tex_path_raw}")

        return materials

    @classmethod
    def _resolve_texture_path(cls, tex_path: str, mtl_dir: str) -> Optional[str]:
        """Resolve texture path to absolute path"""
        tex_path = tex_path.strip('"\'')
        parts = tex_path.split()
        if parts and parts[0].startswith('-'):
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if not part.startswith('-') and '.' in part:
                    tex_path = part
                    break
            else:
                tex_path = parts[-1]

        tex_path = tex_path.replace('\\', '/')

        if os.path.isabs(tex_path):
            if os.path.exists(tex_path):
                return os.path.abspath(tex_path)
            return None

        abs_path = os.path.normpath(os.path.join(mtl_dir, tex_path))
        if os.path.exists(abs_path):
            return abs_path

        basename = os.path.basename(tex_path)
        same_dir_path = os.path.join(mtl_dir, basename)
        if os.path.exists(same_dir_path):
            return os.path.abspath(same_dir_path)

        common_dirs = ['textures', 'tex', 'maps', 'images']
        for subdir in common_dirs:
            subdir_path = os.path.join(mtl_dir, subdir, basename)
            if os.path.exists(subdir_path):
                return os.path.abspath(subdir_path)

        return None

    @classmethod
    def _is_glass_name(cls, name: str) -> bool:
        """Check if name indicates glass material"""
        name_lower = name.lower()
        return name_lower in cls.GLASS_EXACT_NAMES


# ============================================================
# OBJ Splitter
# ============================================================

class OBJSplitter:
    """OBJ file splitter - separates glass, ceiling, wall, floor, and props by material"""

    CEILING_KEYWORDS = ['ceiling', 'roof', 'top', 'sky', 'plafond', 'techo']
    WALL_KEYWORDS = ['wall', 'wand', 'pared', 'mur', 'parede']
    FLOOR_KEYWORDS = ['floor', 'ground', 'boden', 'suelo', 'sol', 'piso']

    def __init__(self, obj_path: str, materials: Dict[str, Dict]):
        self.obj_path = obj_path
        self.materials = materials
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.glass_faces = []
        self.ceiling_faces = []
        self.wall_faces = []
        self.floor_faces = []
        self.prop_faces_by_material = {}

    @classmethod
    def _is_ceiling(cls, mat_name: str) -> bool:
        name_lower = mat_name.lower()
        return any(kw in name_lower for kw in cls.CEILING_KEYWORDS)

    @classmethod
    def _is_wall(cls, mat_name: str) -> bool:
        name_lower = mat_name.lower()
        return any(kw in name_lower for kw in cls.WALL_KEYWORDS)

    @classmethod
    def _is_floor(cls, mat_name: str) -> bool:
        name_lower = mat_name.lower()
        return any(kw in name_lower for kw in cls.FLOOR_KEYWORDS)

    def parse_and_split(self) -> Dict[str, Any]:
        """Parse OBJ and split into category files"""
        current_material = None
        current_type = 'prop'

        with open(self.obj_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                if cmd == 'v' and len(parts) >= 4:
                    self.vertices.append(line)
                elif cmd == 'vn' and len(parts) >= 4:
                    self.normals.append(line)
                elif cmd == 'vt' and len(parts) >= 3:
                    self.texcoords.append(line)
                elif cmd == 'usemtl' and len(parts) >= 2:
                    current_material = parts[1]
                    mat_info = self.materials.get(current_material, {})
                    if mat_info.get('is_glass', False):
                        current_type = 'glass'
                    elif self._is_ceiling(current_material):
                        current_type = 'ceiling'
                    elif self._is_wall(current_material):
                        current_type = 'wall'
                    elif self._is_floor(current_material):
                        current_type = 'floor'
                    else:
                        current_type = 'prop'
                elif cmd == 'f':
                    if current_type == 'glass':
                        self.glass_faces.append(line)
                    elif current_type == 'ceiling':
                        self.ceiling_faces.append(line)
                    elif current_type == 'wall':
                        self.wall_faces.append(line)
                    elif current_type == 'floor':
                        self.floor_faces.append(line)
                    else:
                        if current_material not in self.prop_faces_by_material:
                            self.prop_faces_by_material[current_material] = []
                        self.prop_faces_by_material[current_material].append(line)

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='stereo_split_')
        base_name = os.path.splitext(os.path.basename(self.obj_path))[0]

        self.temp_dir = temp_dir
        paths = {}

        categories = [
            ('glass', self.glass_faces),
            ('ceiling', self.ceiling_faces),
            ('wall', self.wall_faces),
            ('floor', self.floor_faces),
        ]

        for cat_name, faces in categories:
            path = os.path.join(temp_dir, f"{base_name}_{cat_name}.obj")
            self._write_obj(path, faces)
            paths[cat_name] = path

        paths['props'] = {}
        total_prop_faces = 0
        for mat_name, faces in self.prop_faces_by_material.items():
            if faces:
                safe_name = mat_name.replace('/', '_').replace('\\', '_')
                path = os.path.join(temp_dir, f"{base_name}_prop_{safe_name}.obj")
                self._write_obj(path, faces)
                paths['props'][mat_name] = path
                total_prop_faces += len(faces)

        print(f"  [OBJ Split] Glass: {len(self.glass_faces)}, "
              f"Ceiling: {len(self.ceiling_faces)}, Wall: {len(self.wall_faces)}, "
              f"Floor: {len(self.floor_faces)}, Props: {total_prop_faces}")

        return paths

    def _write_obj(self, path: str, faces: List[str]):
        """Write OBJ file"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Split OBJ file\n")
            for v in self.vertices:
                f.write(v + '\n')
            for vn in self.normals:
                f.write(vn + '\n')
            for vt in self.texcoords:
                f.write(vt + '\n')
            for face in faces:
                f.write(face + '\n')

    def get_glass_centroid(self) -> Optional[Tuple[float, float, float]]:
        """
        計算玻璃物體的質心座標 (Scene Coordinates, mm)

        Returns:
            (x, y, z) 質心座標，如果沒有玻璃則返回 None
        """
        if not self.glass_faces:
            return None

        # 解析所有頂點為數組
        vertex_coords = []
        for v in self.vertices:
            parts = v.split()
            if len(parts) >= 4 and parts[0] == 'v':
                vertex_coords.append([
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ])

        if not vertex_coords:
            return None

        vertex_coords = np.array(vertex_coords)

        # 從 glass_faces 提取用到的頂點索引
        used_vertex_indices = set()
        for face in self.glass_faces:
            parts = face.split()
            if parts[0] != 'f':
                continue
            for p in parts[1:]:
                # face 格式: v, v/vt, v/vt/vn, v//vn
                v_idx = p.split('/')[0]
                try:
                    idx = int(v_idx) - 1  # OBJ 索引從 1 開始
                    if 0 <= idx < len(vertex_coords):
                        used_vertex_indices.add(idx)
                except ValueError:
                    continue

        if not used_vertex_indices:
            return None

        # 計算這些頂點的質心
        glass_vertices = vertex_coords[list(used_vertex_indices)]
        centroid = np.mean(glass_vertices, axis=0)

        return tuple(centroid)


# ============================================================
# Scene Builder (path integrator + back-wall LED)
# ============================================================

class SceneBuilder:
    """Mitsuba scene builder"""

    def __init__(self, obj_path: str):
        self.obj_path = str(Path(obj_path).resolve())
        self.mtl_path = self.obj_path.replace('.obj', '.mtl')

        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"OBJ file not found: {self.obj_path}")

        print(f"[SceneBuilder] OBJ: {self.obj_path}")

        self.materials = MTLParser.parse(self.mtl_path)

        splitter = OBJSplitter(self.obj_path, self.materials)
        self.obj_paths = splitter.parse_and_split()
        self.has_glass = len(splitter.glass_faces) > 0
        self.has_ceiling = len(splitter.ceiling_faces) > 0
        self.has_wall = len(splitter.wall_faces) > 0
        self.has_floor = len(splitter.floor_faces) > 0
        self.prop_materials = list(splitter.prop_faces_by_material.keys())
        self.temp_dir = splitter.temp_dir

        # 計算玻璃物體質心（用於相機追蹤）
        self.glass_centroid = splitter.get_glass_centroid()
        if self.glass_centroid:
            print(f"  [Glass] Centroid: X={self.glass_centroid[0]:.1f}, "
                  f"Y={self.glass_centroid[1]:.1f}, Z={self.glass_centroid[2]:.1f} mm")

        # Debug: print OBJ vertex bounds
        self._print_vertex_bounds(splitter.vertices)

    def _print_vertex_bounds(self, vertices: List[str]):
        """Debug: print OBJ vertex coordinate ranges"""
        if not vertices:
            print("  [OBJ Debug] No vertices found!")
            return

        xs, ys, zs = [], [], []
        for v in vertices:
            parts = v.split()
            if len(parts) >= 4 and parts[0] == 'v':
                xs.append(float(parts[1]))
                ys.append(float(parts[2]))
                zs.append(float(parts[3]))

        if not xs:
            print("  [OBJ Debug] No valid vertices found!")
            return

        print(f"  [OBJ Debug] Vertex ranges (OBJ coords, should be mm):")
        print(f"    X: [{min(xs):.1f}, {max(xs):.1f}] mm")
        print(f"    Y: [{min(ys):.1f}, {max(ys):.1f}] mm")
        print(f"    Z: [{min(zs):.1f}, {max(zs):.1f}] mm")

        # After transform: scale 0.001, rotate X -90°
        # 旋轉矩陣效果: (X, Y, Z) → (X, Z, -Y)
        # Mitsuba X = OBJ X * 0.001
        # Mitsuba Y = OBJ Z * 0.001 (高度)
        # Mitsuba Z = -OBJ Y * 0.001 (負深度)
        print(f"  [OBJ Debug] After transform (Mitsuba coords, meters):")
        print(f"    Mitsuba X: [{min(xs)/1000:.3f}, {max(xs)/1000:.3f}] m (左右)")
        print(f"    Mitsuba Y: [{min(zs)/1000:.3f}, {max(zs)/1000:.3f}] m (高度)")
        print(f"    Mitsuba Z: [{-max(ys)/1000:.3f}, {-min(ys)/1000:.3f}] m (深度, 負=前方)")

    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def build(self,
              camera_position: Tuple[float, float, float],
              camera_target: Tuple[float, float, float],
              spp: int,
              seed: int = None) -> Dict:
        """Build complete scene.

        Args:
            camera_position: camera position (Scene coords, mm)
            camera_target: camera lookat target (Scene coords, mm)
            spp: samples per pixel
            seed: random seed
        """
        scene = {
            'type': 'scene',
            'integrator': self._create_integrator(),
            'sensor': self._create_sensor(camera_position, camera_target, spp, seed),
            'led_emitter': self._create_led_light(),
        }

        scene['envmap'] = {
            'type': 'constant',
            'radiance': {
                'type': 'spectrum',
                'value': Config.CEILING_EMITTER_INTENSITY,
            },
        }

        for name, mesh_dict in self._create_meshes():
            scene[name] = mesh_dict

        return scene

    def _create_integrator(self) -> Dict:
        """Create path integrator."""
        return {
            'type': 'path',
            'max_depth': Config.MAX_DEPTH,
        }

    def _create_sensor(self,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       spp: int,
                       seed: int = None) -> Dict:
        """Create camera sensor"""
        pos_m = self._transform_point(position)
        tgt_m = self._transform_point(target)

        # Debug: 顯示座標轉換
        print(f"  [Camera] Scene coords (mm): pos={position}, target={target}")
        print(f"  [Camera] Mitsuba coords (m): pos={pos_m}, target={tgt_m}")
        direction = np.array(tgt_m) - np.array(pos_m)
        direction_norm = direction / np.linalg.norm(direction)
        print(f"  [Camera] View direction: {direction_norm.tolist()}")

        sampler_dict = {
            'type': 'ldsampler',  # Low-discrepancy sampler for variance reduction
            'sample_count': spp,
        }
        if seed is not None:
            sampler_dict['seed'] = seed

        return {
            'type': 'perspective',
            'fov': Config.FOV,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=pos_m,
                target=tgt_m,
                up=[0, 1, 0],  # Mitsuba Y 是向上 (來自 Scene Z)
            ),
            'film': {
                'type': 'hdrfilm',
                'width': Config.WIDTH,
                'height': Config.HEIGHT,
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': sampler_dict,
        }

    def _create_led_light(self) -> Dict:
        """Create back-wall LED area emitter."""
        led_pos_scene = Config.LED_POSITION
        led_target_scene = Config.LED_TARGET

        pos_m = self._transform_point(led_pos_scene)
        tgt_m = self._transform_point(led_target_scene)
        print(f"  [LED] Scene: pos={led_pos_scene}, target={led_target_scene}")
        print(f"  [LED] Mitsuba: pos={pos_m}, target={tgt_m}")

        size_x = mm_to_m(Config.LED_SIZE[0])
        size_y = mm_to_m(Config.LED_SIZE[1])

        emitter_transform = mi.ScalarTransform4f.look_at(
            origin=pos_m,
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2, size_y/2, 1])

        return {
            'type': 'rectangle',
            'to_world': emitter_transform,
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'spectrum',
                    'value': Config.LED_INTENSITY,
                },
            },
        }


    def _create_meshes(self) -> List[Tuple[str, Dict]]:
        """Create split OBJ meshes (with texture support)"""
        meshes = []
        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        # Wall
        wall_path = self.obj_paths.get('wall')
        if self.has_wall and wall_path and os.path.exists(wall_path):
            meshes.append(('mesh_wall', {
                'type': 'obj',
                'filename': wall_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': self._select_wall_bsdf(),
            }))

        # Floor
        floor_path = self.obj_paths.get('floor')
        if self.has_floor and floor_path and os.path.exists(floor_path):
            meshes.append(('mesh_floor', {
                'type': 'obj',
                'filename': floor_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': self._select_floor_bsdf(),
            }))

        # Props (with textures)
        props_dict = self.obj_paths.get('props', {})
        for i, mat_name in enumerate(self.prop_materials):
            prop_path = props_dict.get(mat_name)
            if prop_path and os.path.exists(prop_path):
                meshes.append((f'mesh_prop_{i}', {
                    'type': 'obj',
                    'filename': prop_path,
                    'face_normals': False,
                    'to_world': transform,
                    'bsdf': self._select_material_bsdf(mat_name),
                }))

        # Ceiling (with emitter)
        ceiling_path = self.obj_paths.get('ceiling')
        if self.has_ceiling and ceiling_path and os.path.exists(ceiling_path):
            meshes.append(('mesh_ceiling', {
                'type': 'obj',
                'filename': ceiling_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': self._select_ceiling_bsdf(),
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'spectrum',
                        'value': Config.CEILING_EMITTER_INTENSITY,
                    },
                },
            }))

        # Glass
        glass_path = self.obj_paths.get('glass')
        if self.has_glass and glass_path and os.path.exists(glass_path):
            meshes.append(('mesh_glass', {
                'type': 'obj',
                'filename': glass_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.glass(),
            }))

        return meshes

    def _select_wall_bsdf(self) -> Dict:
        """Select wall BSDF (twosided for interior rendering)"""
        for mat_name, mat_info in self.materials.items():
            if not OBJSplitter._is_wall(mat_name):
                continue
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                return MaterialFactory.twosided(MaterialFactory.diffuse_textured(tex_path))
            color = mat_info.get('color', (0.7, 0.7, 0.7))
            return MaterialFactory.twosided(MaterialFactory.diffuse_rgb(color))
        return MaterialFactory.twosided(MaterialFactory.diffuse(0.7))

    def _select_floor_bsdf(self) -> Dict:
        """Select floor BSDF (twosided for interior rendering)"""
        for mat_name, mat_info in self.materials.items():
            if not OBJSplitter._is_floor(mat_name):
                continue
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                return MaterialFactory.twosided(MaterialFactory.diffuse_textured(tex_path))
            color = mat_info.get('color', (0.5, 0.5, 0.5))
            return MaterialFactory.twosided(MaterialFactory.diffuse_rgb(color))
        return MaterialFactory.twosided(MaterialFactory.diffuse(0.5))

    def _select_material_bsdf(self, mat_name: str) -> Dict:
        """Select prop material BSDF (twosided for interior rendering)"""
        mat_info = self.materials.get(mat_name, {})
        tex_path = mat_info.get('textures', {}).get('diffuse')
        if tex_path and os.path.exists(tex_path):
            print(f"  [BSDF] Using texture: {mat_name} -> {tex_path}")
            return MaterialFactory.twosided(MaterialFactory.diffuse_textured(tex_path))
        color = mat_info.get('color', (0.5, 0.5, 0.5))
        print(f"  [BSDF] Fallback to color: {mat_name} -> RGB{color}")
        return MaterialFactory.twosided(MaterialFactory.diffuse_rgb(color))

    def _select_ceiling_bsdf(self) -> Dict:
        """Select ceiling BSDF (twosided for interior rendering)"""
        for mat_name, mat_info in self.materials.items():
            if not OBJSplitter._is_ceiling(mat_name):
                continue
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                return MaterialFactory.twosided(MaterialFactory.diffuse_textured(tex_path))
        return MaterialFactory.twosided(MaterialFactory.diffuse(0.85))

    def _transform_point(self, point: Tuple[float, float, float]) -> List[float]:
        """
        座標轉換：Scene/OBJ (mm) → Mitsuba (m)

        Scene/OBJ 座標系:
        - X: 左(-) ↔ 右(+)
        - Y: 深度 (0=相機前方, +=遠離相機)
        - Z: 高度 (0=地面, +=向上)

        Mesh Transform: rotate([1,0,0], -90) 繞 X 軸旋轉 -90°
        旋轉矩陣:
        [1,   0,   0]
        [0,   0,   1]
        [0,  -1,   0]

        結果: OBJ (X, Y, Z) → Mitsuba (X, Z, -Y)

        Mitsuba 座標系:
        - X: 左右 (右為正) - 不變
        - Y: 高度 (向上為正) = Scene Z
        - Z: 深度 (向後為正) = -Scene Y

        注意: Mitsuba Z 是負深度，所以場景物體在 Z < 0 的區域
        """
        x, y, z = point
        # Scene (X, Y_depth, Z_height) → Mitsuba (X, Z, -Y)
        return [mm_to_m(x), mm_to_m(z), -mm_to_m(y)]


# ============================================================
# Scene Report Generation
# ============================================================

def generate_scene_report(scene_name: str,
                          left_rgb: np.ndarray,
                          right_rgb: np.ndarray,
                          depth: np.ndarray,
                          disparity: np.ndarray,
                          glass_mask_left: np.ndarray,
                          glass_mask_right: np.ndarray) -> dict:
    """Generate a per-scene quality report."""
    from datetime import datetime

    left_lum = rgb_to_luminance(left_rgb)
    right_lum = rgb_to_luminance(right_rgb)

    report = {
        'scene_name': scene_name,
        'timestamp': datetime.now().isoformat(),
        'render_config': {
            'width': Config.WIDTH,
            'height': Config.HEIGHT,
            'spp': Config.SPP,
            'max_depth': Config.MAX_DEPTH,
        },
        'value_range': {
            'left': {
                'min': float(np.min(left_lum)),
                'max': float(np.max(left_lum)),
                'mean': float(np.mean(left_lum)),
            },
            'right': {
                'min': float(np.min(right_lum)),
                'max': float(np.max(right_lum)),
                'mean': float(np.mean(right_lum)),
            },
        },
    }

    glass_left_bool = (glass_mask_left > 0.5) if glass_mask_left is not None else np.zeros_like(left_lum, dtype=bool)
    glass_right_bool = (glass_mask_right > 0.5) if glass_mask_right is not None else np.zeros_like(right_lum, dtype=bool)

    report['glass_pixels'] = {
        'left': int(np.sum(glass_left_bool)),
        'right': int(np.sum(glass_right_bool)),
        'intersection': int(np.sum(glass_left_bool & glass_right_bool)),
    }

    if depth is not None and np.any(glass_left_bool):
        glass_depth = depth[glass_left_bool]
        valid = glass_depth > 0
        validity_rate = float(np.sum(valid) / glass_depth.size)
        report['glass_depth_validity'] = {
            'glass_pixel_count': int(glass_left_bool.sum()),
            'valid_depth_count': int(valid.sum()),
            'validity_rate': validity_rate,
            'pass': validity_rate >= 0.9,
        }
    else:
        report['glass_depth_validity'] = {
            'glass_pixel_count': int(glass_left_bool.sum()),
            'valid_depth_count': 0,
            'validity_rate': 0.0,
            'pass': False,
        }

    return report


# ============================================================
# Stereo Renderer
# ============================================================

class StereoRenderer:
    """Stereo renderer (single + multi-GPU dispatch)."""

    def __init__(self):
        self.variant = setup_mitsuba()
        self._denoiser = None  # Lazy init

    def _denoise_rgb(self, image: np.ndarray) -> np.ndarray:
        """Apply OptiX denoiser to RGB image (H,W,3)

        Args:
            image: RGB image as numpy array (H,W,3), float32

        Returns:
            Denoised RGB image (H,W,3), float32
        """
        if not Config.DENOISE:
            return image

        # Lazy init denoiser (reuse for same resolution)
        h, w = image.shape[:2]
        if self._denoiser is None or self._denoiser_size != (h, w):
            try:
                self._denoiser = mi.OptixDenoiser(
                    input_size=[w, h],  # width, height
                    albedo=False,
                    normals=False,
                    temporal=False
                )
                self._denoiser_size = (h, w)
                print(f"  [Denoiser] OptixDenoiser initialized ({w}x{h})")
            except Exception as e:
                print(f"  [Denoiser] Failed to init OptixDenoiser: {e}")
                self._denoiser = None
                return image

        # Convert to Mitsuba TensorXf and denoise
        try:
            # Ensure float32 and contiguous
            img_f32 = np.ascontiguousarray(image.astype(np.float32))
            img_mi = mi.TensorXf(img_f32)
            denoised_mi = self._denoiser(img_mi)
            denoised = np.array(denoised_mi).astype(np.float32)
            return denoised
        except Exception as e:
            print(f"  [Denoiser] Denoise failed: {e}")
            return image

    def render_scene(self, obj_path: str, output_dir: str, scene_name: str = None, seed: int = None):
        """Render a single stereo scene."""
        if scene_name is None:
            scene_name = Path(obj_path).stem

        scene_dir = Path(output_dir) / scene_name
        preview_dir = scene_dir / 'preview'
        os.makedirs(preview_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Rendering scene: {scene_name}")
        if seed is not None:
            print(f"  Seed: {seed}")
        print(f"{'='*60}")

        # Build scene first (parse OBJ to obtain glass centroid)
        builder = SceneBuilder(obj_path)

        # Randomize camera pose (conservative jitter)
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        Config.randomize_camera(rng)

        # Apply glass lookat (gentle rotation toward the glass centroid)
        if builder.glass_centroid and Config.GLASS_LOOKAT_ENABLED:
            glass_x = builder.glass_centroid[0]
            Config.apply_glass_lookat(glass_x)
            print(f"[Camera] Glass tracking: centroid_x={glass_x:.1f}mm -> "
                  f"lookat_offset={Config._glass_lookat_x_offset:+.1f}mm")

        cam_offsets = Config.get_camera_offsets()
        if Config.CAMERA_RANDOMIZE:
            total_lookat_x = cam_offsets['lookat_x_offset'] + cam_offsets['glass_lookat_x_offset']
            print(f"[Camera] Jitter: depth={cam_offsets['camera_depth_offset']:+.1f}mm, "
                  f"height={cam_offsets['camera_height_offset']:+.1f}mm, "
                  f"LookAt=({total_lookat_x:+.1f}, {cam_offsets['lookat_z_offset']:+.1f})")

        # Camera positions
        left_pos = Config.left_camera_position()
        right_pos = Config.right_camera_position()
        depth_pos = Config.depth_camera_position()

        left_target = Config.camera_target_for_position(left_pos)
        right_target = Config.camera_target_for_position(right_pos)
        depth_target = Config.camera_target_for_position(depth_pos)

        print(f"[Config] Baseline: {Config.BASELINE}mm, FOV: {Config.FOV}deg, SPP: {Config.SPP}")

        # Debug: show coordinate transformation
        print(f"\n[Coord Debug] Scene -> Mitsuba transformation:")
        print(f"  Left camera:  Scene {left_pos} -> Mitsuba {builder._transform_point(left_pos)}")
        print(f"  Left target:  Scene {left_target} -> Mitsuba {builder._transform_point(left_target)}")
        print(f"  LED position: Scene {Config.LED_POSITION} -> Mitsuba {builder._transform_point(Config.LED_POSITION)}")
        print(f"  LED target:   Scene {Config.LED_TARGET} -> Mitsuba {builder._transform_point(Config.LED_TARGET)}")

        # [1/4] Left camera
        print(f"\n[1/4] Rendering left camera...")
        left_seed = seed if seed is not None else None
        left_rgb = self._render_camera(builder, left_pos, left_target, seed=left_seed)
        gc.collect()

        # [2/4] Right camera
        print(f"\n[2/4] Rendering right camera...")
        right_seed = (seed + 1) if seed is not None else None
        right_rgb = self._render_camera(builder, right_pos, right_target, seed=right_seed)
        gc.collect()

        # Apply OptiX denoiser (if enabled)
        if Config.DENOISE:
            print(f"\n[Denoise] Applying OptiX denoiser...")
            left_rgb = self._denoise_rgb(left_rgb)
            right_rgb = self._denoise_rgb(right_rgb)
            print(f"  [Denoise] Done")

        # [3/4] Depth + glass masks
        print(f"\n[3/4] Rendering depth and glass masks...")
        depth = self._render_depth(builder, depth_pos, depth_target)
        glass_mask_left = self._render_glass_mask(builder, left_pos, left_target)
        glass_mask_right = self._render_glass_mask(builder, right_pos, right_target)

        # [4/4] Compute disparity
        print(f"\n[4/4] Computing disparity...")
        disparity = self._compute_disparity(depth)

        # Save outputs
        print(f"\n[Save] Writing outputs...")
        self._save_outputs(scene_dir, scene_name, left_rgb, right_rgb,
                           depth, disparity, glass_mask_left, glass_mask_right,
                           builder.has_glass, seed)

        builder.cleanup()

        print(f"\n{'='*60}")
        print(f"Scene {scene_name} complete!")
        print(f"Output: {scene_dir}")
        print(f"{'='*60}\n")

    def _render_camera(self,
                       builder: SceneBuilder,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       seed: int = None) -> np.ndarray:
        """Render an RGB camera image and return it as (H, W, 3) float32."""
        if seed is not None:
            scene_dict = builder.build(position, target, Config.SPP, seed)
            scene = mi.load_dict(scene_dict)
            print(f"  [Render] SPP: {Config.SPP}, seed: {seed}")
            image = mi.render(scene, spp=Config.SPP)
            result = np.array(image).astype(np.float32)
        else:
            scene_dict = builder.build(position, target, Config.SPP_PER_BATCH, None)
            scene = mi.load_dict(scene_dict)

            total_spp = Config.SPP
            batch_spp = Config.SPP_PER_BATCH
            n_batches = max(1, total_spp // batch_spp)

            print(f"  [Render] SPP: {total_spp}, batches: {n_batches} x {batch_spp}")

            accumulated = None
            for i in range(n_batches):
                image = mi.render(scene, spp=batch_spp)
                img_np = np.array(image)

                if accumulated is None:
                    accumulated = img_np.astype(np.float64)
                else:
                    accumulated += img_np.astype(np.float64)

                print(f"    Batch {i+1}/{n_batches} done")
                del image
                gc.collect()

            result = (accumulated / n_batches).astype(np.float32)

        print(f"  [Render] Complete, shape: {result.shape}")
        return result

    def _render_depth(self,
                      builder: SceneBuilder,
                      position: Tuple[float, float, float],
                      target: Tuple[float, float, float]) -> np.ndarray:
        """Render depth map"""
        pos_m = builder._transform_point(position)
        tgt_m = builder._transform_point(target)

        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'dd.y:depth',
                'integrator': {
                    'type': 'path',
                    'max_depth': Config.MAX_DEPTH,
                },
            },
            'sensor': {
                'type': 'perspective',
                'fov': Config.FOV,
                'fov_axis': 'x',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=pos_m,
                    target=tgt_m,
                    up=[0, 1, 0],  # Mitsuba Y 是向上 (來自 Scene Z)
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': Config.WIDTH,
                    'height': Config.HEIGHT,
                    'pixel_format': 'luminance',
                    'component_format': 'float32',
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': 64,
                },
            },
        }

        for name, mesh_dict in builder._create_meshes():
            scene_dict[name] = mesh_dict

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=64)
        img_np = np.array(image)

        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth = img_np[:, :, 1]
        elif img_np.ndim == 3:
            depth = img_np[:, :, 0]
        else:
            depth = img_np

        valid = depth > 0
        if valid.any():
            print(f"  [Depth] range: [{depth[valid].min():.3f}, {depth[valid].max():.3f}] m")

        return depth.astype(np.float32)

    def _render_glass_mask(self,
                           builder: SceneBuilder,
                           position: Tuple[float, float, float],
                           target: Tuple[float, float, float]) -> np.ndarray:
        """Render glass region mask"""
        if not builder.has_glass:
            print(f"  [Glass Mask] No glass in scene")
            return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)

        pos_m = builder._transform_point(position)
        tgt_m = builder._transform_point(target)

        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'dd.y:depth',
                'integrator': {
                    'type': 'path',
                    'max_depth': 2,
                },
            },
            'sensor': {
                'type': 'perspective',
                'fov': Config.FOV,
                'fov_axis': 'x',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=pos_m,
                    target=tgt_m,
                    up=[0, 1, 0],  # Mitsuba Y 是向上 (來自 Scene Z)
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': Config.WIDTH,
                    'height': Config.HEIGHT,
                    'pixel_format': 'luminance',
                    'component_format': 'float32',
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': 4,
                },
            },
            'glass_mesh': {
                'type': 'obj',
                'filename': builder.obj_paths.get('glass'),
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.diffuse(0.5),
            },
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=4)
        img_np = np.array(image)

        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth_channel = img_np[:, :, 1]
        elif img_np.ndim == 3:
            depth_channel = img_np[:, :, 0]
        else:
            depth_channel = img_np

        glass_mask = (depth_channel > 0).astype(np.float32)
        pixel_count = int(np.sum(glass_mask))
        pixel_ratio = pixel_count / (Config.WIDTH * Config.HEIGHT)
        print(f"  [Glass Mask] pixels: {pixel_count} ({pixel_ratio*100:.1f}%)")

        return glass_mask

    def _compute_disparity(self, ray_depth: np.ndarray) -> np.ndarray:
        """Compute disparity from depth"""
        fov_rad = np.radians(Config.FOV)
        focal_px = (Config.WIDTH / 2) / np.tan(fov_rad / 2)
        baseline_m = mm_to_m(Config.BASELINE)

        cx, cy = Config.WIDTH / 2, Config.HEIGHT / 2
        y_coords, x_coords = np.meshgrid(
            np.arange(Config.HEIGHT),
            np.arange(Config.WIDTH),
            indexing='ij'
        )
        dx = x_coords - cx
        dy = y_coords - cy

        cos_angle = focal_px / np.sqrt(focal_px**2 + dx**2 + dy**2)
        z_depth = ray_depth * cos_angle

        disparity = np.zeros_like(z_depth)
        valid = z_depth > 0
        disparity[valid] = (baseline_m * focal_px) / z_depth[valid]

        print(f"  [Disparity] focal: {focal_px:.1f}px, "
              f"range: [{disparity[valid].min():.1f}, {disparity[valid].max():.1f}]px")

        return disparity.astype(np.float32)

    def _save_exr(self, image: np.ndarray, path: Path):
        """Save EXR file"""
        if image.ndim == 3 and image.shape[2] == 3:
            bitmap = mi.Bitmap(image.astype(np.float32), mi.Bitmap.PixelFormat.RGB)
        else:
            if image.ndim == 3:
                image = image[:, :, 0]
            bitmap = mi.Bitmap(image.astype(np.float32))
        bitmap.write(str(path))
        print(f"    -> {path}")

    def _save_png(self, image: np.ndarray, path: Path, vmin: float = None, vmax: float = None):
        """Save PNG (percentile normalize + gamma)."""
        if image.ndim == 3 and image.shape[2] >= 3:
            rgb = image[:, :, :3].copy()
        else:
            rgb = image.copy()

        # Handle empty/invalid images
        rgb = np.maximum(rgb, 0)
        if rgb.max() <= 0:
            if rgb.ndim == 3:
                output = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
            else:
                output = np.zeros(rgb.shape, dtype=np.uint8)
            cv2.imwrite(str(path), output)
            print(f"    -> {path}")
            return

        max_val = np.percentile(rgb, 99.5)
        if max_val <= 0:
            max_val = 1.0

        normalized = rgb / max_val
        normalized = np.clip(normalized, 0, 1)

        # Gamma correction
        normalized = np.power(normalized, 1.0 / 2.2)

        # Convert to 8-bit
        output = (normalized * 255).astype(np.uint8)

        if output.ndim == 3:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), output)
        print(f"    -> {path}")

    def _save_outputs(self,
                      scene_dir: Path,
                      scene_name: str,
                      left_rgb: np.ndarray,
                      right_rgb: np.ndarray,
                      depth: np.ndarray,
                      disparity: np.ndarray,
                      glass_mask_left: np.ndarray,
                      glass_mask_right: np.ndarray,
                      has_glass: bool,
                      seed: int = None):
        """Save scene outputs (EXR + masks + previews + params + QA report)."""
        preview_dir = scene_dir / 'preview'
        os.makedirs(preview_dir, exist_ok=True)

        # EXR files
        self._save_exr(left_rgb, scene_dir / 'left.exr')
        self._save_exr(right_rgb, scene_dir / 'right.exr')
        self._save_exr(depth, scene_dir / 'depth.exr')
        self._save_exr(disparity, scene_dir / 'disparity.exr')

        # Per-view glass masks
        cv2.imwrite(str(scene_dir / 'glass_mask_left.png'),
                    (glass_mask_left * 255).astype(np.uint8))
        cv2.imwrite(str(scene_dir / 'glass_mask_right.png'),
                    (glass_mask_right * 255).astype(np.uint8))
        print(f"    -> {scene_dir / 'glass_mask_left.png'}")
        print(f"    -> {scene_dir / 'glass_mask_right.png'}")

        # Intersection mask (training / strict eval)
        glass_mask_intersection = ((glass_mask_left > 0.5) & (glass_mask_right > 0.5)).astype(np.float32)
        cv2.imwrite(str(scene_dir / 'glass_mask.png'),
                    (glass_mask_intersection * 255).astype(np.uint8))
        print(f"    -> {scene_dir / 'glass_mask.png'} (intersection)")

        # Preview PNGs
        if Config.SAVE_PREVIEW:
            left_lum = rgb_to_luminance(left_rgb)
            right_lum = rgb_to_luminance(right_rgb)
            combined = np.concatenate([left_lum.flatten(), right_lum.flatten()])
            vmax = np.percentile(combined, 99.5)
            self._save_png(left_rgb, preview_dir / 'left.png', 0.0, vmax)
            self._save_png(right_rgb, preview_dir / 'right.png', 0.0, vmax)

        # Params JSON
        left_lum = rgb_to_luminance(left_rgb)
        right_lum = rgb_to_luminance(right_rgb)
        params = {
            'format': 'mitsuba_stereo',
            'width': Config.WIDTH,
            'height': Config.HEIGHT,
            'baseline_mm': Config.BASELINE,
            'fov_deg': Config.FOV,
            'spp': Config.SPP,
            'seed': seed,
            'has_glass': has_glass,
            'glass_pixels_left': int(np.sum(glass_mask_left > 0.5)),
            'glass_pixels_right': int(np.sum(glass_mask_right > 0.5)),
            'left_range': [float(left_lum.min()), float(left_lum.max())],
            'right_range': [float(right_lum.min()), float(right_lum.max())],
            'camera': {
                'coordinate_system': 'camera_centric',
                'depth': Config.CAMERA_DEPTH,
                'height': Config.CAMERA_HEIGHT,
                'baseline': Config.BASELINE,
                'randomize': Config.CAMERA_RANDOMIZE,
                **Config.get_camera_offsets(),
            },
        }
        with open(scene_dir / 'params.json', 'w') as f:
            json.dump(params, f, indent=2)
        print(f"    -> {scene_dir / 'params.json'}")

        # Per-scene quality report
        report = generate_scene_report(
            scene_name=scene_name,
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            depth=depth,
            disparity=disparity,
            glass_mask_left=glass_mask_left,
            glass_mask_right=glass_mask_right,
        )
        report_path = scene_dir / f'{scene_name}_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"    -> {report_path}")


# ============================================================
# Multi-GPU Support (Dynamic Work Queue with multiprocessing)
# ============================================================

def gpu_worker(gpu_id: int,
               task_queue,
               result_dict,
               output_dir: str,
               spp: int,
               glass_type: str,
               seed: Optional[int]):
    """
    Worker process for a specific GPU - pulls tasks from queue dynamically.

    Args:
        gpu_id: GPU index (sets CUDA_VISIBLE_DEVICES)
        task_queue: multiprocessing.Queue with scene paths
        result_dict: Manager().dict() for collecting results
        output_dir: Output directory
        spp: Samples per pixel
        glass_type: Glass material type
        seed: Optional MC seed
    """
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Apply config in worker process
    Config.SPP = spp
    Config.GLASS_TYPE = glass_type

    # Initialize renderer (will init CUDA on this GPU)
    renderer = StereoRenderer()
    rendered_count = 0

    while True:
        try:
            # Non-blocking get with timeout
            scene_path_str = task_queue.get(timeout=1)
        except:
            # Queue empty, check if really done
            if task_queue.empty():
                break
            continue

        if scene_path_str is None:  # Poison pill
            break

        scene_path = Path(scene_path_str)
        start_time = time.time()

        print(f"  [GPU {gpu_id}] Starting {scene_path.name}...")

        try:
            renderer.render_scene(str(scene_path), output_dir, seed=seed)
            elapsed = time.time() - start_time
            print(f"  [GPU {gpu_id}] {scene_path.name} done in {elapsed:.1f}s")
            result_dict[scene_path.name] = ('success', elapsed)
            rendered_count += 1
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            print(f"  [GPU {gpu_id}] {scene_path.name} FAILED: {error_msg}")
            result_dict[scene_path.name] = ('failed', elapsed, error_msg)

            # 寫入錯誤 log
            import traceback
            log_path = Path(output_dir) / 'render_errors.log'
            with open(log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Scene: {scene_path.name}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Time: {elapsed:.1f}s\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")

    print(f"  [GPU {gpu_id}] Worker finished. Rendered {rendered_count} scenes.")


def run_multi_gpu(args, scenes: List[Path]):
    """
    Multi-GPU rendering with dynamic work queue using multiprocessing.

    Mechanism:
    1. Main process creates work queue (multiprocessing.Queue)
    2. Spawns N worker processes, each bound to different GPU
    3. Workers dynamically pull tasks from shared queue
    4. Continues until queue is empty (poison pill received)

    This is more efficient than static partitioning because:
    - Scenes have different complexity (some render faster)
    - Automatic load balancing across GPUs
    - No external dependencies (uses built-in multiprocessing)
    """
    from multiprocessing import Process, Queue, Manager

    num_gpus = args.num_gpus
    total_scenes = len(scenes)

    print(f"\n[Multi-GPU] Dynamic work queue mode")
    print(f"  Total scenes: {total_scenes}")
    print(f"  GPUs: {num_gpus}")
    print(f"  SPP: {args.spp}")

    # Create task queue and add all scenes
    task_queue = Queue()
    for scene_path in scenes:
        task_queue.put(str(scene_path))

    # Add poison pills (one per worker)
    for _ in range(num_gpus):
        task_queue.put(None)

    # Shared dict for results
    manager = Manager()
    result_dict = manager.dict()

    # Spawn worker processes
    workers = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_dict, args.output,
                  args.spp, args.glass_type, args.seed)
        )
        p.start()
        workers.append(p)
        print(f"[Launch] GPU {gpu_id} PID: {p.pid}")

    print(f"\n[Multi-GPU] Waiting for all GPUs to complete...")

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Summarize results
    success_count = 0
    fail_count = 0
    total_time = 0
    failed_scenes = []

    for scene_name, result in result_dict.items():
        if result[0] == 'success':
            success_count += 1
            total_time += result[1]
        else:
            fail_count += 1
            total_time += result[1]
            error_msg = result[2] if len(result) > 2 else 'Unknown'
            failed_scenes.append((scene_name, error_msg))

    print(f"\n[Multi-GPU] All GPUs completed!")
    print(f"  Success: {success_count}/{total_scenes}")
    print(f"  Failed: {fail_count}")
    print(f"  Total render time: {total_time/60:.1f} minutes")

    # 儲存失敗清單
    if failed_scenes:
        failed_list_path = Path(args.output) / 'failed_scenes.txt'
        with open(failed_list_path, 'w') as f:
            f.write(f"# Failed scenes: {fail_count}\n")
            f.write(f"# Total: {total_scenes}\n\n")
            for scene_name, error_msg in failed_scenes:
                f.write(f"{scene_name}\t{error_msg[:100]}\n")
        print(f"  Failed list saved to: {failed_list_path}")
        print(f"  Error details in: {Path(args.output) / 'render_errors.log'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mitsuba Stereo Renderer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render single scene
  python stereo_renderer.py --scene ./scenes/scene_001.obj --output ./output

  # Render all scenes in directory
  python stereo_renderer.py --input_dir ./scenes --output ./output

  # Render with specific seed (for MC variance analysis)
  python stereo_renderer.py --input_dir ./scenes --output ./output --seed 42

  # Render with custom SPP
  python stereo_renderer.py --input_dir ./scenes --output ./output --spp 2048

  # Multi-GPU rendering (4 GPUs)
  python stereo_renderer.py --input_dir ./scenes --output ./output --num_gpus 4

  # Re-render failed scenes from list
  python stereo_renderer.py --input_dir ./scenes --scene_list failed_list.txt --output ./output --num_gpus 8
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene', type=str, help='Single OBJ scene')
    input_group.add_argument('--input_dir', type=str, help='OBJ scene directory')

    parser.add_argument('--scene_list', type=str, default=None,
                        help='Text file with scene names to render (one per line, requires --input_dir)')

    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--spp', type=int, default=Config.SPP, help=f'SPP (default: {Config.SPP})')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for MC sampler')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum number of scenes')
    parser.add_argument('--skip', type=int, default=0, help='Skip first N scenes')
    parser.add_argument('--glass_type', type=str, choices=['thin', 'thick'], default='thick',
                        help='Glass material type')

    # Multi-GPU arguments
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs for parallel rendering (default: 1)')

    # Camera randomization
    parser.add_argument('--no-camera-jitter', action='store_true',
                        help='Disable camera position randomization')

    args = parser.parse_args()

    # Apply config
    Config.SPP = args.spp
    Config.GLASS_TYPE = args.glass_type
    Config.CAMERA_RANDOMIZE = not args.no_camera_jitter

    # Collect scenes
    if args.scene:
        scenes = [Path(args.scene)]
    else:
        input_dir = Path(args.input_dir)

        # 如果有 scene_list，只渲染清單中的場景
        if args.scene_list:
            with open(args.scene_list, 'r') as f:
                scene_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            scenes = []
            for name in scene_names:
                # 支援多種格式: scene_0001, scene_0001.obj, 或完整路徑
                if name.endswith('.obj'):
                    obj_path = input_dir / name
                else:
                    # 嘗試目錄結構: input_dir/scene_name/scene_name.obj
                    obj_path = input_dir / name / f"{name}.obj"
                    if not obj_path.exists():
                        # 或者直接: input_dir/scene_name.obj
                        obj_path = input_dir / f"{name}.obj"
                if obj_path.exists():
                    scenes.append(obj_path)
                else:
                    print(f"[Warning] Scene not found: {name}")
        else:
            # 原本的邏輯：掃描所有 OBJ
            all_objs = sorted(input_dir.glob('*.obj'))
            scenes = [f for f in all_objs
                      if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]

            # 也搜尋子目錄 (scene_name/scene_name.obj)
            if not scenes:
                for subdir in sorted(input_dir.iterdir()):
                    if subdir.is_dir() and subdir.name.startswith('scene_'):
                        obj_file = subdir / f"{subdir.name}.obj"
                        if obj_file.exists():
                            scenes.append(obj_file)

        if args.skip > 0:
            scenes = scenes[args.skip:]
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]

    print(f"[Mitsuba Stereo Renderer]")
    print(f"  SPP: {Config.SPP}")
    print(f"  Seed: {args.seed}")
    print(f"  Glass type: {Config.GLASS_TYPE}")
    print(f"  Resolution: {Config.WIDTH}x{Config.HEIGHT}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Camera jitter: {'ON' if Config.CAMERA_RANDOMIZE else 'OFF'}")
    print()

    os.makedirs(args.output, exist_ok=True)

    # Multi-GPU mode
    if args.num_gpus > 1:
        run_multi_gpu(args, scenes)
        sys.exit(0)

    # Single GPU mode
    renderer = StereoRenderer()
    start_time = time.time()

    for i, scene_path in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] Processing {scene_path.name}...")
        try:
            renderer.render_scene(str(scene_path), args.output, seed=args.seed)
        except Exception as e:
            print(f"  [ERROR] Failed to render {scene_path.name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All done! {len(scenes)} scenes in {elapsed/60:.1f} minutes")
    print(f"Output: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

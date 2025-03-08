import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from typing import List
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 3D軸のスケールを等しくするための補助関数
def set_axes_equal(ax):
    """3DプロットのX, Y, Z軸のスケールを等しくする"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# 2D,3D用のデータクラス定義
@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Edge2D:
    start: Point2D
    end: Point2D

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Edge3D:
    start: Point3D
    end: Point3D

# ---------- IrimoyaRoof クラス（追加） ----------
class IrimoyaRoof:
    def __init__(self, base_width: float, base_depth: float, ridge_length: float, 
                 ridge_height: float, lower_hip_height: float, hip_roof_top_width: float):
        """入母屋屋根を生成するクラス
        
        Parameters:
            base_width: 建物の横幅（X方向）
            base_depth: 建物の奥行（Y方向）
            ridge_length: 棟の長さ（切妻部分の幅）
            ridge_height: 棟の最高点の高さ
            lower_hip_height: 寄棟部分の最高点の高さ
            hip_roof_top_width: 寄棟部分の屋根の台形の上辺の長さ
        """
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_length = ridge_length
        self.ridge_height = ridge_height
        self.lower_hip_height = lower_hip_height
        self.hip_roof_top_width = hip_roof_top_width
        self.calculate_points()
    
    def calculate_points(self):
        """3D空間上の各頂点を計算"""
        half_width = self.base_width / 2
        half_depth = self.base_depth / 2
        half_ridge = self.ridge_length / 2
        half_hip_top = self.hip_roof_top_width / 2
        
        # 基礎となる矩形の頂点
        self.points = {
            'base_fl': Point3D(-half_width, 0, 0),  # front left
            'base_fr': Point3D(half_width, 0, 0),   # front right
            'base_bl': Point3D(-half_width, 0, self.base_depth),  # back left
            'base_br': Point3D(half_width, 0, self.base_depth),   # back right
        }
        
        # 寄棟部分の上部頂点（台形の上辺の頂点）
        self.points.update({
            'hip_fl': Point3D(-half_hip_top, self.lower_hip_height, 0),
            'hip_fr': Point3D(half_hip_top, self.lower_hip_height, 0),
            'hip_bl': Point3D(-half_hip_top, self.lower_hip_height, self.base_depth),
            'hip_br': Point3D(half_hip_top, self.lower_hip_height, self.base_depth),
        })
        
        # 棟の頂点（最上部）
        upper_height = self.ridge_height - self.lower_hip_height
        self.points.update({
            'ridge_f': Point3D(0, self.ridge_height, 0),  # front
            'ridge_b': Point3D(0, self.ridge_height, self.base_depth),  # back
        })

    def get_faces(self) -> List[List[Point3D]]:
        """入母屋屋根の面を構成する頂点リストを取得"""
        pts = self.points
        
        return [
            # 寄棟部分（4面）
            [pts['base_fl'], pts['base_fr'], pts['hip_fr'], pts['hip_fl']],  # 前面
            [pts['base_bl'], pts['base_br'], pts['hip_br'], pts['hip_bl']],  # 背面
            [pts['base_fl'], pts['base_bl'], pts['hip_bl'], pts['hip_fl']],  # 左側面
            [pts['base_fr'], pts['base_br'], pts['hip_br'], pts['hip_fr']],  # 右側面
            
            # 切妻部分（4面）
            [pts['hip_fl'], pts['hip_fr'], pts['ridge_f']],  # 前面三角形
            [pts['hip_bl'], pts['hip_br'], pts['ridge_b']],  # 背面三角形
            [pts['hip_fl'], pts['hip_bl'], pts['ridge_b'], pts['ridge_f']],  # 左側面
            [pts['hip_fr'], pts['hip_br'], pts['ridge_b'], pts['ridge_f']],  # 右側面
        ]

    def get_edges(self) -> List[Edge3D]:
        """入母屋屋根を構成するエッジのリストを取得"""
        pts = self.points
        edges = []
        
        # 基礎の矩形
        base_edges = [
            (pts['base_fl'], pts['base_fr']),
            (pts['base_fr'], pts['base_br']),
            (pts['base_br'], pts['base_bl']),
            (pts['base_bl'], pts['base_fl']),
        ]
        edges.extend([Edge3D(start, end) for start, end in base_edges])
        
        # 寄棟部分への垂直エッジ
        vertical_edges = [
            (pts['base_fl'], pts['hip_fl']),
            (pts['base_fr'], pts['hip_fr']),
            (pts['base_bl'], pts['hip_bl']),
            (pts['base_br'], pts['hip_br']),
        ]
        edges.extend([Edge3D(start, end) for start, end in vertical_edges])
        
        # 寄棟部分の上辺
        hip_top_edges = [
            (pts['hip_fl'], pts['hip_fr']),
            (pts['hip_fr'], pts['hip_br']),
            (pts['hip_br'], pts['hip_bl']),
            (pts['hip_bl'], pts['hip_fl']),
        ]
        edges.extend([Edge3D(start, end) for start, end in hip_top_edges])
        
        # 切妻部分への稜線
        ridge_edges = [
            (pts['hip_fl'], pts['ridge_f']),
            (pts['hip_fr'], pts['ridge_f']),
            (pts['hip_bl'], pts['ridge_b']),
            (pts['hip_br'], pts['ridge_b']),
            (pts['ridge_f'], pts['ridge_b']),  # 棟線
        ]
        edges.extend([Edge3D(start, end) for start, end in ridge_edges])
        
        return edges

    def get_front_elevation(self) -> List[Edge2D]:
        """正面立面図を構成するエッジのリストを取得"""
        half_width = self.base_width / 2
        half_hip_top = self.hip_roof_top_width / 2
        
        # 台形部分の頂点
        base_left = Point2D(-half_width, 0)
        base_right = Point2D(half_width, 0)
        hip_left = Point2D(-half_hip_top, self.lower_hip_height)
        hip_right = Point2D(half_hip_top, self.lower_hip_height)
        
        # 三角形部分の頂点
        ridge_top = Point2D(0, self.ridge_height)
        
        edges = []
        # 台形部分
        trapezoid_edges = [
            (base_left, base_right),
            (base_right, hip_right),
            (hip_right, hip_left),
            (hip_left, base_left),
        ]
        edges.extend([Edge2D(start, end) for start, end in trapezoid_edges])
        
        # 三角形部分
        triangle_edges = [
            (hip_left, ridge_top),
            (hip_right, ridge_top),
        ]
        edges.extend([Edge2D(start, end) for start, end in triangle_edges])
        
        return edges

    def get_side_elevation(self) -> List[Edge2D]:
        """側面立面図を構成するエッジのリストを取得"""
        half_depth = self.base_depth / 2
        half_ridge = self.ridge_length / 2
        
        # 台形部分の頂点
        base_front = Point2D(-half_depth, 0)
        base_back = Point2D(half_depth, 0)
        hip_front = Point2D(-half_ridge, self.lower_hip_height)
        hip_back = Point2D(half_ridge, self.lower_hip_height)
        
        # 矩形部分の頂点
        ridge_height = self.ridge_height
        rect_top_front = Point2D(-half_ridge, ridge_height)
        rect_top_back = Point2D(half_ridge, ridge_height)
        
        edges = []
        # 台形部分
        trapezoid_edges = [
            (base_front, base_back),
            (base_back, hip_back),
            (hip_back, hip_front),
            (hip_front, base_front),
        ]
        edges.extend([Edge2D(start, end) for start, end in trapezoid_edges])
        
        # 矩形部分
        rectangle_edges = [
            (hip_front, rect_top_front),
            (rect_top_front, rect_top_back),
            (rect_top_back, hip_back),
        ]
        edges.extend([Edge2D(start, end) for start, end in rectangle_edges])
        
        return edges



# ---------- FlatRoof クラス（追加） ----------
class FlatRoof:
    def __init__(self, base_width: float, base_depth: float, ridge_height: float):
        """
        Parameters:
            base_width: 建物・屋根の横幅
            base_depth: 建物の奥行
            ridge_height: 屋根の高さ（平らな面の厚みとして表現）
        """
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height
        self.calculate_points()

    def calculate_points(self):
        half_width = self.base_width / 2
        # 2D・3D用の主要点を算出
        self.front_bottom_left = Point3D(-half_width, 0, 0)
        self.front_bottom_right = Point3D(half_width, 0, 0)
        self.back_bottom_left = Point3D(-half_width, 0, self.base_depth)
        self.back_bottom_right = Point3D(half_width, 0, self.base_depth)
        self.front_top_left = Point3D(-half_width, self.ridge_height, 0)
        self.front_top_right = Point3D(half_width, self.ridge_height, 0)
        self.back_top_left = Point3D(-half_width, self.ridge_height, self.base_depth)
        self.back_top_right = Point3D(half_width, self.ridge_height, self.base_depth)

    def get_edges(self) -> List[Edge3D]:
        edges = []
        # 前面（Front face）
        edges.append(Edge3D(self.front_bottom_left, self.front_bottom_right))
        edges.append(Edge3D(self.front_bottom_left, self.front_top_left))
        edges.append(Edge3D(self.front_bottom_right, self.front_top_right))
        edges.append(Edge3D(self.front_top_left, self.front_top_right))
        # 背面（Back face）
        edges.append(Edge3D(self.back_bottom_left, self.back_bottom_right))
        edges.append(Edge3D(self.back_bottom_left, self.back_top_left))
        edges.append(Edge3D(self.back_bottom_right, self.back_top_right))
        edges.append(Edge3D(self.back_top_left, self.back_top_right))
        # 前後を連結する辺
        edges.append(Edge3D(self.front_bottom_left, self.back_bottom_left))
        edges.append(Edge3D(self.front_bottom_right, self.back_bottom_right))
        edges.append(Edge3D(self.front_top_left, self.back_top_left))
        edges.append(Edge3D(self.front_top_right, self.back_top_right))
        return edges

    def get_front_elevation(self) -> List[Edge2D]:
        half_width = self.base_width / 2
        # 正面図：矩形
        return [
            Edge2D(Point2D(-half_width, 0), Point2D(half_width, 0)),             # 下辺
            Edge2D(Point2D(-half_width, 0), Point2D(-half_width, self.ridge_height)),  # 左縦辺
            Edge2D(Point2D(-half_width, self.ridge_height), Point2D(half_width, self.ridge_height)),  # 上辺
            Edge2D(Point2D(half_width, self.ridge_height), Point2D(half_width, 0))   # 右縦辺
        ]

    def get_side_elevation(self) -> List[Edge2D]:
        # 側面図：矩形
        return [
            Edge2D(Point2D(0, 0), Point2D(self.base_depth, 0)),                  # 下辺
            Edge2D(Point2D(0, 0), Point2D(0, self.ridge_height)),                  # 前縦辺
            Edge2D(Point2D(0, self.ridge_height), Point2D(self.base_depth, self.ridge_height)),  # 上辺
            Edge2D(Point2D(self.base_depth, self.ridge_height), Point2D(self.base_depth, 0))       # 後縦辺
        ]
    def get_faces(self) -> List[List[Point3D]]:
        # 陸屋根は水平な屋根面として、上面の矩形（front_top_left, front_top_right, back_top_right, back_top_left）を返す
        return [[
            self.front_top_left,
            self.front_top_right,
            self.back_top_right,
            self.back_top_left
        ]]

class PyramidRoof:
    def __init__(self, base_width: float, base_depth: float, ridge_height: float):
        """
        Parameters:
            base_width: 建物・屋根の横幅
            base_depth: 建物の奥行
            ridge_height: ピークの高さ
        """
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height
        self.calculate_points()

    def calculate_points(self):
        half_width = self.base_width / 2
        self.points = {
            'front_left': Point3D(-half_width, 0, 0),
            'front_right': Point3D(half_width, 0, 0),
            'back_right': Point3D(half_width, 0, self.base_depth),
            'back_left': Point3D(-half_width, 0, self.base_depth)
        }
        self.peak = Point3D(0, self.ridge_height, self.base_depth/2)

    def get_edges(self) -> List[Edge3D]:
        p = self.points
        return [
            Edge3D(p['front_left'], p['front_right']),
            Edge3D(p['front_right'], p['back_right']),
            Edge3D(p['back_right'], p['back_left']),
            Edge3D(p['back_left'], p['front_left']),
            Edge3D(p['front_left'], self.peak),
            Edge3D(p['front_right'], self.peak),
            Edge3D(p['back_right'], self.peak),
            Edge3D(p['back_left'], self.peak),
        ]

    def get_faces(self) -> List[List[Point3D]]:
        p = self.points
        return [
            [p['front_left'], p['front_right'], self.peak],
            [p['front_right'], p['back_right'], self.peak],
            [p['back_right'], p['back_left'], self.peak],
            [p['back_left'], p['front_left'], self.peak],
        ]

    def get_front_elevation(self) -> List[Edge2D]:
        # 正面図：三角形（底辺 = base_width, 高さ = ridge_height）
        return [
            Edge2D(Point2D(-self.base_width/2, 0), Point2D(self.base_width/2, 0)),
            Edge2D(Point2D(-self.base_width/2, 0), Point2D(0, self.ridge_height)),
            Edge2D(Point2D(0, self.ridge_height), Point2D(self.base_width/2, 0))
        ]

    def get_side_elevation(self) -> List[Edge2D]:
        # 側面図：三角形（底辺 = base_depth, 高さ = ridge_height）
        return [
            Edge2D(Point2D(-self.base_depth/2, 0), Point2D(self.base_depth/2, 0)),
            Edge2D(Point2D(-self.base_depth/2, 0), Point2D(0, self.ridge_height)),
            Edge2D(Point2D(0, self.ridge_height), Point2D(self.base_depth/2, 0))
        ]


# ---------- ShedRoof クラス（追加） ----------
class ShedRoof:
    def __init__(self, base_width: float, base_depth: float, ridge_height: float):
        """
        Parameters:
            base_width: 建物・屋根の横幅
            base_depth: 建物の奥行
            ridge_height: 屋根の傾斜による高さ（低い側は0、高い側はridge_height）
        """
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height
        self.calculate_points()

    def calculate_points(self):
        half_width = self.base_width / 2
        # 片流れ屋根は、前面エッジが高さ0、背面エッジがridge_heightとする
        self.points = {
            'front_left': Point3D(-half_width, 0, 0),
            'front_right': Point3D(half_width, 0, 0),
            'back_left': Point3D(-half_width, self.ridge_height, self.base_depth),
            'back_right': Point3D(half_width, self.ridge_height, self.base_depth)
        }

    def get_edges(self) -> List[Edge3D]:
        # 4点で1枚の平面（屋根面）を形成
        return [
            Edge3D(self.points['front_left'], self.points['front_right']),
            Edge3D(self.points['front_right'], self.points['back_right']),
            Edge3D(self.points['back_right'], self.points['back_left']),
            Edge3D(self.points['back_left'], self.points['front_left'])
        ]

    def get_faces(self) -> List[List[Point3D]]:
        return [[
            self.points['front_left'],
            self.points['front_right'],
            self.points['back_right'],
            self.points['back_left']
        ]]

    def get_front_elevation(self) -> List[Edge2D]:
        # 正面図：直角三角形（頂点：A=(-base_width/2,0), B=(-base_width/2, ridge_height), C=(base_width/2, 0)）
        return [
            Edge2D(Point2D(-self.base_width/2, 0), Point2D(-self.base_width/2, self.ridge_height)),
            Edge2D(Point2D(-self.base_width/2, self.ridge_height), Point2D(self.base_width/2, 0)),
            Edge2D(Point2D(self.base_width/2, 0), Point2D(-self.base_width/2, 0))
        ]

    def get_side_elevation(self) -> List[Edge2D]:
        # 側面図：長方形（底辺=base_depth, 高さ=ridge_height）
        return [
            Edge2D(Point2D(0, 0), Point2D(self.base_depth, 0)),
            Edge2D(Point2D(self.base_depth, 0), Point2D(self.base_depth, self.ridge_height)),
            Edge2D(Point2D(self.base_depth, self.ridge_height), Point2D(0, self.ridge_height)),
            Edge2D(Point2D(0, self.ridge_height), Point2D(0, 0))
        ]

# ------------- HipRoof クラス-------------
class HipRoof:
    def __init__(self, base_width: float, base_depth: float, ridge_length: float, ridge_height: float):
        """
        Parameters:
            base_width: width of the building (horizontal extent)
            base_depth: depth of the building
            ridge_length: length of the ridge line
            ridge_height: height of the ridge (vertical distance from eaves to ridge)
        """
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_length = ridge_length
        self.ridge_height = ridge_height
        self.calculate_points()

    def calculate_points(self):
        half_width = self.base_width / 2
        half_ridge = self.ridge_length / 2
        ridge_offset = (self.base_depth - self.ridge_length) / 2
        self.points = {
            'front_left': Point3D(-half_width, 0, 0),
            'front_right': Point3D(half_width, 0, 0),
            'back_left': Point3D(-half_width, 0, self.base_depth),
            'back_right': Point3D(half_width, 0, self.base_depth),
            'ridge_front': Point3D(0, self.ridge_height, ridge_offset),
            'ridge_back': Point3D(0, self.ridge_height, ridge_offset + self.ridge_length),
        }

    def get_edges(self) -> List[Edge3D]:
        edges = []
        # Front triangle edges
        edges.append(Edge3D(self.points['front_left'], self.points['ridge_front']))
        edges.append(Edge3D(self.points['ridge_front'], self.points['front_right']))
        edges.append(Edge3D(self.points['front_right'], self.points['front_left']))
        # Back triangle edges
        edges.append(Edge3D(self.points['back_left'], self.points['ridge_back']))
        edges.append(Edge3D(self.points['ridge_back'], self.points['back_right']))
        edges.append(Edge3D(self.points['back_right'], self.points['back_left']))
        # Left trapezoid edges
        edges.append(Edge3D(self.points['front_left'], self.points['back_left']))
        edges.append(Edge3D(self.points['back_left'], self.points['ridge_back']))
        edges.append(Edge3D(self.points['ridge_back'], self.points['ridge_front']))
        edges.append(Edge3D(self.points['ridge_front'], self.points['front_left']))
        # Right trapezoid edges
        edges.append(Edge3D(self.points['front_right'], self.points['back_right']))
        edges.append(Edge3D(self.points['back_right'], self.points['ridge_back']))
        edges.append(Edge3D(self.points['ridge_back'], self.points['ridge_front']))
        edges.append(Edge3D(self.points['ridge_front'], self.points['front_right']))
        return edges

    def get_faces(self) -> List[List[Point3D]]:
        return [
            [self.points['front_left'], self.points['ridge_front'], self.points['front_right']],  # Front roof panel
            [self.points['back_left'], self.points['ridge_back'], self.points['back_right']],       # Back roof panel
            [self.points['front_left'], self.points['back_left'], self.points['ridge_back'], self.points['ridge_front']],  # Left roof panel
            [self.points['front_right'], self.points['back_right'], self.points['ridge_back'], self.points['ridge_front']],  # Right roof panel
        ]

    def get_front_elevation(self) -> List[Edge2D]:
        return [
            Edge2D(Point2D(-self.base_width / 2, 0), Point2D(0, self.ridge_height)),
            Edge2D(Point2D(0, self.ridge_height), Point2D(self.base_width / 2, 0)),
        ]

    # 修正: 側面図を台形（閉じた形）として描画する
    def get_side_elevation(self) -> List[Edge2D]:
        left_edge_x = (self.base_depth - self.ridge_length) / 2
        right_edge_x = left_edge_x + self.ridge_length
        return [
            Edge2D(Point2D(0, 0), Point2D(left_edge_x, self.ridge_height)),         # 左斜面
            Edge2D(Point2D(left_edge_x, self.ridge_height), Point2D(right_edge_x, self.ridge_height)),  # 上辺
            Edge2D(Point2D(right_edge_x, self.ridge_height), Point2D(self.base_depth, 0)),  # 右斜面
            Edge2D(Point2D(self.base_depth, 0), Point2D(0, 0)),                      # 下辺（閉じるため）
        ]


# 切妻屋根クラス（GableRoof）
class GableRoof:
    def __init__(self, base_width: float, base_depth: float, ridge_height: float):
        """
        Parameters:
          base_width: 建物・屋根の横幅
          base_depth: 建物の奥行
          ridge_height: 棟の高さ（軒先から棟までの垂直距離）
        """
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height
        self.calculate_points()

    def calculate_points(self):
        """切妻屋根の主要な点を計算"""
        half_width = self.base_width / 2
        self.points = {
            'front_left': Point3D(-half_width, 0, 0),
            'front_right': Point3D(half_width, 0, 0),
            'back_left': Point3D(-half_width, 0, self.base_depth),
            'back_right': Point3D(half_width, 0, self.base_depth),
            'front_peak': Point3D(0, self.ridge_height, 0),
            'back_peak': Point3D(0, self.ridge_height, self.base_depth)
        }

    def get_edges(self) -> List[Edge3D]:
        """
        屋根パネルを構成するエッジを返す。
        左右それぞれの屋根パネルと棟の連結エッジを含む。
        """
        edges = []
        # 左側パネル
        edges.append(Edge3D(self.points['front_left'], self.points['front_peak']))
        edges.append(Edge3D(self.points['front_peak'], self.points['back_peak']))
        edges.append(Edge3D(self.points['back_peak'], self.points['back_left']))
        edges.append(Edge3D(self.points['back_left'], self.points['front_left']))
        # 右側パネル
        edges.append(Edge3D(self.points['front_right'], self.points['front_peak']))
        edges.append(Edge3D(self.points['front_peak'], self.points['back_peak']))
        edges.append(Edge3D(self.points['back_peak'], self.points['back_right']))
        edges.append(Edge3D(self.points['back_right'], self.points['front_right']))
        # 棟の共通エッジ
        edges.append(Edge3D(self.points['front_peak'], self.points['back_peak']))
        return edges

    def get_faces(self) -> List[List[Point3D]]:
        """
        屋根パネル（面）のリストを返す。
        左右のパネルそれぞれを1枚の面（四角形）として表現。
        """
        return [
            [self.points['front_left'], self.points['front_peak'], self.points['back_peak'], self.points['back_left']],
            [self.points['front_right'], self.points['front_peak'], self.points['back_peak'], self.points['back_right']]
        ]

    def get_front_elevation(self) -> List[Edge2D]:
        """
        正面図用のエッジを返す（左右の斜面のみ）。
        """
        return [
            Edge2D(Point2D(-self.base_width/2, 0), Point2D(0, self.ridge_height)),
            Edge2D(Point2D(0, self.ridge_height), Point2D(self.base_width/2, 0))
        ]

    def get_side_elevation(self) -> List[Edge2D]:
        """
        側面図用のエッジ（底辺・高さ・奥行を示す矩形）を返す。
        """
        return [
            Edge2D(Point2D(0, 0), Point2D(self.base_depth, 0)),  # 下辺
            Edge2D(Point2D(self.base_depth, 0), Point2D(self.base_depth, self.ridge_height)),  # 右側縦辺
            Edge2D(Point2D(self.base_depth, self.ridge_height), Point2D(0, self.ridge_height)),  # 上辺
            Edge2D(Point2D(0, self.ridge_height), Point2D(0, 0))  # 左側縦辺
        ]

# GUIアプリケーション
class RoofVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Roof Visualizer")
        
        # パラメータ入力用フレーム
        input_frame = ttk.LabelFrame(root, text="Parameters", padding="10")
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # 屋根タイプ選択のコンボボックス（既存部分を以下のように変更）
        ttk.Label(input_frame, text="Roof Type:").grid(row=0, column=0, sticky="w")
        self.roof_type = tk.StringVar(value="kiritsuma")
        roof_combo = ttk.Combobox(input_frame, 
                                textvariable=self.roof_type,
                                values=["kiritsuma", "yosemune", "katanagare", "hougyou", "roku", "irimoya"],
                                state="readonly")
        roof_combo.grid(row=0, column=1, padx=5)
        roof_combo.bind('<<ComboboxSelected>>', lambda e: self.update_visualization())

        # 共通パラメータ
        ttk.Label(input_frame, text="Base Width:").grid(row=1, column=0, sticky="w")
        self.base_width_var = tk.StringVar(value="10")
        base_width_spin = ttk.Spinbox(input_frame, from_=1, to=100, increment=1,
                                      textvariable=self.base_width_var, command=self.update_visualization, width=10)
        base_width_spin.grid(row=1, column=1, padx=5)
        base_width_spin.bind('<Return>', lambda e: self.update_visualization())
        
        ttk.Label(input_frame, text="Base Depth:").grid(row=2, column=0, sticky="w")
        self.base_depth_var = tk.StringVar(value="8")
        base_depth_spin = ttk.Spinbox(input_frame, from_=1, to=100, increment=1,
                                      textvariable=self.base_depth_var, command=self.update_visualization, width=10)
        base_depth_spin.grid(row=2, column=1, padx=5)
        base_depth_spin.bind('<Return>', lambda e: self.update_visualization())
        
        # Hip Roof 用パラメータ【追加】（gable時は無効）
        ttk.Label(input_frame, text="Ridge Length:").grid(row=3, column=0, sticky="w")
        self.ridge_length_var = tk.StringVar(value="5")
        self.ridge_length_spin = ttk.Spinbox(input_frame, from_=1, to=100, increment=1,
                                             textvariable=self.ridge_length_var, command=self.update_visualization, width=10)
        self.ridge_length_spin.grid(row=3, column=1, padx=5)
        self.ridge_length_spin.bind('<Return>', lambda e: self.update_visualization())
        
        ttk.Label(input_frame, text="Ridge Height:").grid(row=4, column=0, sticky="w")
        self.ridge_height_var = tk.StringVar(value="5")
        ridge_height_spin = ttk.Spinbox(input_frame, from_=1, to=100, increment=1,
                                        textvariable=self.ridge_height_var, command=self.update_visualization, width=10)
        ridge_height_spin.grid(row=4, column=1, padx=5)
        ridge_height_spin.bind('<Return>', lambda e: self.update_visualization())

        # 既存のパラメータ入力（例：Base Width, Base Depth, Ridge Height, Ridge Length）の下に追加
        ttk.Label(input_frame, text="Lower Hip Height:").grid(row=5, column=0, sticky="w")
        self.lower_hip_height_var = tk.StringVar(value="3")
        self.lower_hip_height_spin = ttk.Spinbox(
            input_frame,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.lower_hip_height_var,
            command=self.update_visualization,
            width=10
        )
        self.lower_hip_height_spin.grid(row=5, column=1, padx=5)
        self.lower_hip_height_spin.bind('<Return>', lambda e: self.update_visualization())

        # 入母屋屋根用の追加パラメータ
        ttk.Label(input_frame, text="Hip Roof Top Width:").grid(row=6, column=0, sticky="w")
        self.hip_roof_top_width_var = tk.StringVar(value="4")
        self.hip_roof_top_width_spin = ttk.Spinbox(
            input_frame,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.hip_roof_top_width_var,
            command=self.update_visualization,
            width=10
        )
        self.hip_roof_top_width_spin.grid(row=6, column=1, padx=5)
        self.hip_roof_top_width_spin.bind('<Return>', lambda e: self.update_visualization())

        # Canvas 設定など（既存コードと同様）
        self.fig = plt.Figure(figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=5)
        
        self.update_visualization()

    def draw_2d(self, edges: List[Edge2D], title: str, ax: plt.Axes):
        ax.clear()
        ax.set_title(title)
        for edge in edges:
            ax.plot([edge.start.x, edge.end.x],
                    [edge.start.y, edge.end.y],
                    'b-', linewidth=2)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def draw_3d(self, edges: List[Edge3D], faces: List[List[Point3D]], ax: plt.Axes):
        """ 3Dエッジと面を描画 """
        ax.clear()
        ax.set_title('3D View')

        # 三角柱（切妻部分）と寄棟部分を個別に表示
        for edge in edges:
            ax.plot([edge.start.x, edge.end.x],
                    [edge.start.z, edge.end.z],
                    [edge.start.y, edge.end.y],
                    'k-', linewidth=2)

        for face in faces:
            verts = [[(p.x, p.z, p.y) for p in face]]
            poly = Poly3DCollection(verts, facecolors='blue', alpha=0.3)
            ax.add_collection3d(poly)

        ax.set_xlabel('X (Width)')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Height)')
        ax.view_init(elev=30, azim=45)  # 角度を変更してわかりやすく
        set_axes_equal(ax)

    def update_visualization(self):
        try:
            base_width = float(self.base_width_var.get())
            base_depth = float(self.base_depth_var.get())
            ridge_height = float(self.ridge_height_var.get())
            roof_type = self.roof_type.get()
            
            if roof_type == "kiritsuma":
                roof = GableRoof(base_width, base_depth, ridge_height)
                self.ridge_length_spin.state(['disabled'])
                self.lower_hip_height_spin.state(['disabled'])
                self.hip_roof_top_width_spin.state(['disabled'])
            elif roof_type == "yosemune":
                ridge_length = float(self.ridge_length_var.get())
                roof = HipRoof(base_width, base_depth, ridge_length, ridge_height)
                self.ridge_length_spin.state(['!disabled'])
                self.lower_hip_height_spin.state(['disabled'])
                self.hip_roof_top_width_spin.state(['disabled'])
            elif roof_type == "katanagare":
                roof = ShedRoof(base_width, base_depth, ridge_height)
                self.ridge_length_spin.state(['disabled'])
                self.lower_hip_height_spin.state(['disabled'])
                self.hip_roof_top_width_spin.state(['disabled'])
            elif roof_type == "hougyou":
                roof = PyramidRoof(base_width, base_depth, ridge_height)
                self.ridge_length_spin.state(['disabled']) 
                self.lower_hip_height_spin.state(['disabled'])
                self.hip_roof_top_width_spin.state(['disabled'])
            elif roof_type == "roku":
                roof = FlatRoof(base_width, base_depth, ridge_height)
                self.ridge_length_spin.state(['disabled'])
                self.lower_hip_height_spin.state(['disabled'])
                self.hip_roof_top_width_spin.state(['disabled'])
            elif roof_type == "irimoya":
                ridge_length = float(self.ridge_length_var.get())
                lower_hip_height = float(self.lower_hip_height_var.get())
                hip_roof_top_width = float(self.hip_roof_top_width_var.get())

                roof = IrimoyaRoof(base_width, base_depth, ridge_length, ridge_height, lower_hip_height, hip_roof_top_width)

                self.ridge_length_spin.state(['!disabled'])
                self.lower_hip_height_spin.state(['!disabled'])
                self.hip_roof_top_width_spin.state(['!disabled'])
                self.lower_hip_height_spin.state(['!disabled'])

            edges_3d = roof.get_edges()
            faces = roof.get_faces()
            front_edges = roof.get_front_elevation()
            side_edges = roof.get_side_elevation()
            
            self.fig.clear()
            ax1 = self.fig.add_subplot(131)
            ax2 = self.fig.add_subplot(132)
            ax3 = self.fig.add_subplot(133, projection='3d')
            
            max_dim = max(base_width, base_depth, ridge_height)
            ax1.set_xlim(-max_dim * 1.2, max_dim * 1.2)
            ax1.set_ylim(-max_dim * 0.2, max_dim * 1.2)
            ax2.set_xlim(-max_dim * 0.2, base_depth * 1.2)
            ax2.set_ylim(-max_dim * 0.2, ridge_height * 1.2)
            
            self.draw_2d(front_edges, 'Front Elevation', ax1)
            self.draw_2d(side_edges, 'Side Elevation', ax2)
            self.draw_3d(edges_3d, faces, ax3)
            
            self.fig.tight_layout()
            self.canvas.draw()
        except ValueError as e:
            print(f"Error: {e}")

def main():
    root = tk.Tk()
    app = RoofVisualizerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()

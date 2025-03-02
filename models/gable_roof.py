from models.base_roof import BaseRoof
from utils.data_classes import Vertex3D, VertexTop, VertexFront, VertexSide, Edge2D, Face
from utils.expansion import compute_expanded_face
from typing import List


class GlobalRoof(BaseRoof):
    """切妻屋根クラス"""

    def calculate_points(self):
        """ 3D 頂点の計算 """
        hw = self.base_width / 2
        self.vertices = {
            'A': Vertex3D(-hw, 0, 0),
            'B': Vertex3D(hw, 0, 0),
            'C': Vertex3D(0, self.ridge_height, 0),
            'D': Vertex3D(0, self.ridge_height, self.base_depth),
            'E': Vertex3D(-hw, 0, self.base_depth),
            'F': Vertex3D(hw, 0, self.base_depth)
        }

    def get_top_view(self) -> List[Edge2D]:
        """ 屋根の平面図 """
        return [
            Edge2D((self.vertices['A'].x, self.vertices['A'].z),
                   (self.vertices['B'].x, self.vertices['B'].z)),
            Edge2D((self.vertices['B'].x, self.vertices['B'].z),
                   (self.vertices['F'].x, self.vertices['F'].z)),
            Edge2D((self.vertices['F'].x, self.vertices['F'].z),
                   (self.vertices['E'].x, self.vertices['E'].z)),
            Edge2D((self.vertices['E'].x, self.vertices['E'].z),
                   (self.vertices['A'].x, self.vertices['A'].z)),
            Edge2D((self.vertices['C'].x, self.vertices['C'].z),
                   (self.vertices['D'].x, self.vertices['D'].z))  # ridge
        ]
    
    def get_front_elevation(self) -> List[Edge2D]:
        """ 屋根の前面図 """
        return [
            Edge2D((self.vertices['A'].x, self.vertices['A'].y),
                   (self.vertices['C'].x, self.vertices['C
                                                        '].y)),
            Edge2D((self.vertices['B'].x, self.vertices['B'].y),
                   (self.vertices['F'].x, self.vertices['F'].y)),
            Edge2D((self.vertices['F'].x, self.vertices['F'].y),
                   (self.vertices['E'].x, self.vertices['E'].y)),
            Edge2D((self.vertices['E'].x, self.vertices['E'].y),
                   (self.vertices['A'].x, self.vertices['A'].y)),
            Edge2D((self.vertices['C'].x, self.vertices['C'].y),
                   (self.vertices['D'].x, self.vertices['D'].y))  # ridge
        ]
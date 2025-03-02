from abc import ABC, abstractmethod
from utils.data_classes import Vertex, Edge, Face
from utils.expansion import compute_expanded_face
from typing import List


class BaseRoof(ABC):
    """ 屋根の基底クラス """
    def __init__(self, base_width:float, base_depth:float, ridge_height: float):
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height
        self.calculate_points()

    @abstractmethod
    def calculate_points(self):
        pass

    @abstractmethod
    def get_front_elevation(self) -> List[Edge]:
        pass

    @abstractmethod
    def get_side_elevation(self) -> List[Edge]:
        pass

    @abstractmethod
    def get_top_view(self) -> List[Edge]:
        pass

    @abstractmethod
    def get_expanded_view(self) -> List[Face]:
        pass

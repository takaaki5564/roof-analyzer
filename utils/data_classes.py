from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Vertex3D:
    x: float  # x and z are the coordinates of the vertex
    y: float  # y is the height of the vertex
    z: float


@dataclass
class VertexTop:
    x: float
    z: float


@dataclass
class VertexFront:
    x: float
    y: float


@dataclass
class VertexSide:
    z: float
    y: float


@dataclass
class Edge2D:
    start: Tuple[float, float]  # (x,y) or (x,z) or (z,y)
    end: Tuple[float, float]


@dataclass
class Face:
    vertices: List[VertexTop]
    slop_angle: float  # degree
    slop_dir: Tuple[float, float]  # (dx, dz)
    referece_vertex: Vertex3D  # base vertex in expansion

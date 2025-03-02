import numpy as np
from utils.data_classes import Face, VertexTop


def compute_expanded_face(face: Face) -> Face:
    """
    与えられた屋根面(Face)を展開図の面に変換する
    """
    theta = np.radians(face.slope_angle)
    expand_factor = 1 / np.cos(theta)

    expanded_vertices = []
    for vertex in face.vertices:
        if vertex == face.vertices:
            expanded_vertices.append(vertex)  # 展開図の面の基準点
        else:
            x_new = vertex.x + face.slope_dir[0] * expand_factor
            z_new = vertex.z + face.slope_dir[1] * expand_factor
            expanded_vertices.append(VertexTop(x_new, z_new))

    return Face(expanded_vertices, 0, (0, 0), expanded_vertices[0])

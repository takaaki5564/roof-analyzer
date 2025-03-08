import numpy as np
import math
from enum import Enum
from typing import List, Dict, Tuple, Set, Union, Optional


class NodeType(Enum):
    RIDGE = "ridge"
    EAVES = "eaves"


class EdgeType(Enum):
    RIDGE = "ridge"
    HIP_RIDGE = "hip_ridge"
    VALLEY = "valley"
    EAVES = "eaves"
    GABLE = "gable"
    WINDOW = "window"
    WALL = "wall"


class Node:
    def __init__(self, id: str, x: float, y: float, z: float, attributes: Dict = None):
        self.id = id
        self.position = np.array([x, y, z], dtype=float)
        self.attributes = attributes or {}

    def __repr__(self):
        return f"Node({self.id}, pos={self.position}, attrs={[attr.value for attr in self.attributes]})"

    def project_xy(self) -> np.ndarray:
        return np.array([self.position[0], self.position[1]])

    def project_zy(self) -> np.ndarray:
        return np.array([self.position[2], self.position[1]])

    def project_xz(self) -> np.ndarray:
        return np.array([self.position[0], self.position[2]])


class Edge:
    def __init__(self, id: str, start: Node, end: Node, attributes: Dict = None):
        self.id = id
        self.start = start
        self.end = end
        self.attributes = attributes or []

    def __repr__(self):
        return f"Edge({self.id}, start={self.start.id}, end={self.end.id}, attrs={[attr.value for attr in self.attributes]})"

    def get_nodes(self) -> Tuple[Node, Node]:
        return self.start, self.end


class Face:
    def __init__(self, id: str, nodes: List[Node], edges: List[Edge], attributes: Dict = None):
        self.id = id
        self.nodes = nodes
        self.edges = edges
        self.attributes = attributes or {}

    def __repr__(self):
        return f"Face({self.id}, nodes={[node.id for node in self.nodes]}, edges={[edge.id for edge in self.edges]}, attrs={self.attributes})"

    def calculate_normal(self) -> np.ndarray:
        """ Calcuate the normal of the face """
        if len(self.nodes) < 3:
            raise ValueError("Face must have at least 3 nodes to calculate normal")

        v1 = self.nodes[1].position - self.nodes[0].position
        v2 = self.nodes[2].position - self.nodes[0].position
        normal = np.cross(v1, v2)

        # Normalize
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal /= normal_length

        return normal

    def calculate_tilt_angle(self) -> float:
        """ Calculate the tilt angle of the face """
        normal = self.calculate_normal()

        # Calculate the angle of roof from the horizontal plane
        horizontal_plane = np.array([0, 1, 0])
        cos_angle = np.abs(np.dot(normal, horizontal_plane))
        angle_rad = math.acos(cos_angle)

        tilt_angle = math.degrees(angle_rad)

        return tilt_angle

    def calculate_tilt_vector(self) -> np.ndarray:
        """ Calculate maximum tilt vector of the face """
        normal = self.calculate_normal()

        horizontal_component = np.array([normal[0], 0, normal[2]])

        # Noramalize
        length = np.linalg.norm(horizontal_component)
        if length > 0:
            horizontal_component /= length

        return horizontal_component

    def calculate_centroid(self) -> np.ndarray:
        """ Calculate centroid of face """
        if not self.nodes:
            return np.zeros(3)

        centroid = np.mean([node.position for node in self.nodes], axis=0)
        return centroid

    def calculate_face_area(self) -> float:
        if len(self.nodes) < 3:
            return 0.0

        base_point = self.nodes[0].position
        area = 0.0

        # Calculate area by dividing face into triangles
        for i in range(1, len(self.nodes) - 1):
            v1 = self.nodes[i].position - base_point
            v2 = self.nodes[i + 1].position - base_point
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))

        return area


class ProjectedNode:
    """ Node projected on a 2D plane """
    def __init__(self, id: str, position: np.ndarray, original_node: Node):
        self.id = id
        self.position = position  # 2D position
        self.original_node = original_node  # Original 3D node
        self.attributes = original_node.attributes.copy()

    def __repr__(self):
        return f"ProjectedNode({self.id}, pos={self.position}, original={self.original_node.id}, attrs={[attr.value for attr in self.attributes]})"


class ProjectedEdge:
    """ Edge projected on a 2D plane """
    def __init__(self, id: str, start: ProjectedNode, end: ProjectedNode, original_edge: Edge):
        self.id = id
        self.start = start
        self.end = end
        self.original_edge = original_edge
        self.attributes = original_edge.attributes.copy()

    def __repr__(self):
        return f"ProjectedEdge({self.id}, start={self.start.id}, end={self.end.id}, original={self.original_edge.id}, attrs={[attr.value for attr in self.attributes]})"


class ProjectedGraph:
    """ Graph projected on a specified 2D plane """
    def __init__(self, nodes: List[ProjectedNode], edges: List[ProjectedEdge], projection_type: str):
        self.nodes = nodes
        self.edges = edges
        self.projection_type = projection_type  # front, side, top

    def __repr__(self):
        return f"ProjectedGraph(nodes={[node.id for node in self.nodes]}, edges={[edge.id for edge in self.edges]})"

    def to_dict(self) -> Dict:
        """ Convert graph to use on Graph Matching Network """
        nodes_data = []
        for node in self.nodes:
            node_data = {
                "id": node.id,
                "position": node.position.tolist(),
                "attributes": node.attributes
            }
            nodes_data.append(node_data)

        edges_data = []
        for edge in self.edges:
            edge_data = {
                "id": edge.id,
                "start": edge.start.id,
                "end": edge.end.id,
                "attributes": edge.attributes
            }
            edges_data.append(edge_data)

        return {
            "projection_type": self.projection_type,
            "nodes": nodes_data,
            "edges": edges_data
        }


class BaseRoof:
    """ Base class for roof types """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.faces = []

        self.front_view = None  # XY-plane
        self.side_view = None  # ZY-plane
        self.top_view = None  # XZ-plane

    def _generate_structure(self):
        raise NotImplementedError

    def _generate_projections(self):
        self._generate_front_view()
        self._generate_side_view()
        self._generate_top_view()

    def _generate_front_view(self):
        """ Generate front view projection """
        # Project nodes and edges on XY-plane
        projected_nodes = {}
        # Project nodes which are duplicated in the same position
        node_at_position = {}

        for node in self.nodes:
            pos_2d = node.project_xy()
            pos_tuple = tuple(pos_2d)

            # If node is duplicated, select smaller Z
            if pos_tuple in node_at_position:
                existing_node = node_at_position[pos_tuple]
                if node.position[2] < existing_node.position[2]:
                    proj_node = ProjectedNode(node.id, pos_2d, node)
                    projected_nodes[node.id] = proj_node
                    node_at_position[pos_tuple] = node
            else:
                proj_node = ProjectedNode(node.id, pos_2d, node)
                projected_nodes[node.id] = proj_node
                node_at_position[pos_tuple] = node

        # Project edges
        projected_edges = []

        for edge in self.edges:
            if edge.start.id in projected_nodes and edge.end.id in projected_nodes:
                proj_edge = ProjectedEdge(
                    edge.id,
                    projected_nodes[edge.start.id],
                    projected_nodes[edge.end.id],
                    edge
                )
                projected_edges.append(proj_edge)

        # Create ProjectedGraph object
        self.front_view = ProjectedGraph(
            list(projected_nodes.values()),
            projected_edges,
            "front"
        )

    def _generate_side_view(self):
        """ Generate side view projection """
        # Project nodes and edges on ZY-plane
        projected_nodes = {}
        # Project nodes which are duplicated in the same position
        node_at_position = {}

        for node in self.nodes:
            pos_2d = node.project_zy()
            pos_tuple = tuple(pos_2d)

            # If node is duplicated, select smaller X
            if pos_tuple in node_at_position:
                existing_node = node_at_position[pos_tuple]
                if node.position[0] < existing_node.position[0]:
                    proj_node = ProjectedNode(node.id, pos_2d, node)
                    projected_nodes[node.id] = proj_node
                    node_at_position[pos_tuple] = node
            else:
                proj_node = ProjectedNode(node.id, pos_2d, node)
                projected_nodes[node.id] = proj_node
                node_at_position[pos_tuple] = node

        # Project edges
        projected_edges = []

        for edge in self.edges:
            if edge.start.id in projected_nodes and edge.end.id in projected_nodes:
                proj_edge = ProjectedEdge(
                    edge.id,
                    projected_nodes[edge.start.id],
                    projected_nodes[edge.end.id],
                    edge
                )
                projected_edges.append(proj_edge)

        # Create ProjectedGraph object
        self.side_view = ProjectedGraph(
            list(projected_nodes.values()),
            projected_edges,
            "side"
        )

    def _generate_top_view(self):
        """ Generate top view projection """
        # Project nodes and edges on XZ-plane
        projected_nodes = {}

        for node in self.nodes:
            pos_2d = node.project_xz()
            proj_node = ProjectedNode(node.id, pos_2d, node)
            projected_nodes[node.id] = proj_node

        # Project edges
        projected_edges = []

        for edge in self.edges:
            proj_edge = ProjectedEdge(
                edge.id,
                projected_nodes[edge.start.id],
                projected_nodes[edge.end.id],
                edge
            )
            projected_edges.append(proj_edge)

        # Create ProjectedGraph object
        self.top_view = ProjectedGraph(
            list(projected_nodes.values()),
            projected_edges,
            "top"
        )

    def get_front_view(self) -> Dict:
        return self.front_view.to_dict()

    def get_side_view(self) -> Dict:
        return self.side_view.to_dict()

    def get_top_view(self) -> Dict:
        return self.top_view.to_dict()

    def get_3d_structure(self) -> Dict:
        """ Get 3D structure in dictionary format """
        nodes_data = []
        for node in self.nodes:
            nodes_data.append({
                "id": node.id,
                "position": node.position.tolist(),
                "attributes": node.attributes
            })

        edges_data = []
        for edge in self.edges:
            edges_data.append({
                "id": edge.id,
                "start": edge.start.id,
                "end": edge.end.id,
                "attributes": edge.attributes
            })

        face_data = []
        for face in self.faces:
            face_data.append({
                "id": face.id,
                "node_ids": [node.id for node in face.nodes],
                "edge_ids": [edge.id for edge in face.edges],
                "attributes": face.attributes
            })

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "faces": face_data
        }

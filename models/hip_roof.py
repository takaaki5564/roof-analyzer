from roof_models.base import BaseRoof, Node, Edge, Face, NodeType, EdgeType
import numpy as np
import math


class HipRoof(BaseRoof):
    """ HipRoof (Yosemune-Yane) class """

    def __init__(self, base_width: float, base_depth: float, ridge_height: float, ridge_length: float):
        """
        Initialize a HipRoof (Yosemune-Yane) object

        Args:
        base_width: float: Width of the roofbase (X-axis)
        base_depth: float: Depth of the roofbase (Z-axis)
        ridge_height: float: Height of the ridge (Y-axis)
        ridge_length: float: Length of the ridge (Z-axis)
        """
        super().__init__()
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height
        self.ridge_length = ridge_length

        # Generate 3D structure
        self._generate_structure()

        # Generate 2D projections
        self._generate_projections()

    def _generate_structure(self):
        bw = self.base_width
        bd = self.base_depth
        rh = self.ridge_height
        rl = self.ridge_length
        hbw = self.base_width / 2
        hbd = self.base_depth / 2
        dd = (bd - rl) / 2

        # Generate nodes
        node_a = Node("A", 0, 0, 0, {"node_type": NodeType.EAVES})
        node_b = Node("B", bw, 0, 0, {"node_type": NodeType.EAVES})
        node_c = Node("C", hbw, rh, dd, {"node_type": NodeType.RIDGE})
        node_d = Node("D", 0, 0, bd, {"node_type": NodeType.EAVES})
        node_e = Node("E", bw, 0, bd, {"node_type": NodeType.EAVES})
        node_f = Node("F", hbw, rh, dd+rl, {"node_type": NodeType.RIDGE})

        self.nodes = [node_a, node_b, node_c, node_d, node_e, node_f]

        # Calculate angles
        theta1 = math.degrees(math.atan2(rh, hbd))  # Angle for triangular faces
        theta2 = math.degrees(math.atan2(rh, hbw))  # Angle for trapezoidal faces

        # Generate edges
        edge_ab = Edge("AB", node_a, node_b, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": 0,
            "length": np.linalg.norm(node_b.position - node_a.position)
            })
        edge_ca = Edge("CA", node_c, node_a, {
            "edge_type": EdgeType.HIP_RIDGE,
            "slope_angle": theta1,
            "length": np.linalg.norm(node_a.position - node_c.position)
            })
        edge_cb = Edge("CB", node_c, node_b, {
            "edge_type": EdgeType.HIP_RIDGE,
            "slope_angle": theta1,
            "length": np.linalg.norm(node_b.position - node_c.position)
            })
        edge_cf = Edge("CF", node_c, node_f, {
            "edge_type": EdgeType.RIDGE,
            "slope_angle": 0,
            "length": np.linalg.norm(node_f.position - node_c.position)
            })
        edge_df = Edge("DF", node_d, node_f, {
            "edge_type": EdgeType.HIP_RIDGE,
            "slope_angle": theta1,
            "length": np.linalg.norm(node_f.position - node_d.position)
            })
        edge_ef = Edge("EF", node_e, node_f, {
            "edge_type": EdgeType.HIP_RIDGE,
            "slope_angle": theta1,
            "length": np.linalg.norm(node_f.position - node_e.position)
            })
        edge_de = Edge("DE", node_d, node_e, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": 0,
            "length": np.linalg.norm(node_e.position - node_d.position)
            })
        edge_ad = Edge("AD", node_a, node_d, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": 0,
            "length": np.linalg.norm(node_d.position - node_a.position)
            })
        edge_be = Edge("BE", node_b, node_e, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": 0,
            "length": np.linalg.norm(node_e.position - node_b.position)
            })

        self.edges = [edge_ab, edge_ca, edge_cb, edge_cf, edge_df, edge_ef, edge_de, edge_ad, edge_be]

        # Generate faces
        # Trapezoidal face ACFD
        face_acfd = Face(
            "ACFD",
            [node_a, node_c, node_f, node_d],
            [edge_ca, edge_cf, edge_df, edge_ad],
            {"face_type": "roof_plane", "slope_angle": theta2}
        )

        # Trapezoidal face BCFE
        face_bcfe = Face(
            "BCFE",
            [node_b, node_c, node_f, node_e],
            [edge_cb, edge_cf, edge_ef, edge_be],
            {"face_type": "roof_plane", "slope_angle": theta2}
        )

        # Triangular face ABC
        face_abc = Face(
            "ABC",
            [node_a, node_b, node_c],
            [edge_ab, edge_ca, edge_cb],
            {"face_type": "roof_plane", "slope_angle": theta1}
        )

        # Triangular face DEF
        face_def = Face(
            "DEF",
            [node_d, node_e, node_f],
            [edge_de, edge_ef, edge_df],
            {"face_type": "roof_plane", "slope_angle": theta1}
        )

        self.faces = [face_acfd, face_bcfe, face_abc, face_def]

        # Additional attributes for faces
        for face in self.faces:
            # Tilt angle
            face.attributes["tilt_angle"] = face.calculate_tilt_angle()

            # Maximum tilt vector
            face.attributes["tilt_vector"] = face.calculate_tilt_vector().tolist()

            # Centroid
            face.attributes["centroid"] = face.calculate_centroid().tolist()

            # Area size
            face.attributes["area"] = face.calculate_face_area()

            face.attributes["roof_stype"] = "hip"
            face.attributes["roof_parameters"] = {
                "base_width": bw,
                "base_depth": bd,
                "ridge_height": rh,
                "ridge_length": rl
            }


if __name__ == "__main__":
    # Example usage with ridge_length parameter
    roof = HipRoof(10, 15, 7, 8)

    # Get 3D structure
    structure_3d = roof.get_3d_structure()
    print("structure_3d")
    print(f"num of nodes: {len(structure_3d['nodes'])}")
    print(f"num of edges: {len(structure_3d['edges'])}")
    print(f"num of faces: {len(structure_3d['faces'])}")

    # Get node attributes
    print("Node attributes:")
    print(structure_3d['nodes'][0]['attributes'])

    # Get edge attributes
    print("Edge attributes:")
    print(structure_3d['edges'][0]['attributes'])

    # Get face attributes
    print("Face attributes:")
    print(structure_3d['faces'][0]['attributes'])

    # Get tilt angle from horizontal
    print("Face tilt angles:")
    for face in structure_3d['faces']:
        print(f"Face id {face['id']}: tilt angle: {face['attributes']['tilt_angle']} degree")
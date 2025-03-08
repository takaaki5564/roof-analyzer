from models.base_roof import BaseRoof, Node, Edge, Face, NodeType, EdgeType
import numpy as np
import math


class GableRoof(BaseRoof):
    """ GableRoof (Kiritsuma-Yane) class """

    def __init__(self, base_width: float, base_depth: float, ridge_height: float):
        """
        Initialize a GableRoof (Kiritsuma-Yane) object

        Args:
        base_width: float: Width of the roofbase (X-axis)
        base_depth: float: Depth of the roofbase (Z-axis)
        ridge_height: float: Height of the ridge (Y-axis)
        """
        super().__init__()
        self.base_width = base_width
        self.base_depth = base_depth
        self.ridge_height = ridge_height

        # Generate 3D structure
        self._generate_structure()

        # Generate 2D projections
        self._generate_projections()

    def _generate_structure(self):
        bw = self.base_width
        bd = self.base_depth
        rh = self.ridge_height
        hbw = self.base_width / 2

        # Generate nodes
        node_a = Node("A", 0, 0, 0, {"node_type": NodeType.EAVES})
        node_b = Node("B", bw, 0, 0, {"node_type": NodeType.EAVES})
        node_c = Node("C", hbw, rh, 0, {"node_type": NodeType.RIDGE})
        node_d = Node("D", 0, 0, bd, {"node_type": NodeType.EAVES})
        node_e = Node("E", bw, 0, bd, {"node_type": NodeType.EAVES})
        node_f = Node("F", hbw, rh, bd, {"node_type": NodeType.RIDGE})

        self.nodes = [node_a, node_b, node_c, node_d, node_e, node_f]

        # Generate edges
        roof_angle = math.degrees(math.atan2(rh, hbw))

        edge_ac = Edge("AC", node_a, node_c, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": roof_angle,
            "length": np.linalg.norm(node_c.position - node_a.position)
            })
        edge_bc = Edge("BC", node_b, node_c, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": roof_angle,
            "length": np.linalg.norm(node_c.position - node_b.position)
            })
        edge_cf = Edge("CF", node_c, node_f, {
            "edge_type": EdgeType.RIDGE,
            "slope_angle": 0,
            "length": np.linalg.norm(node_f.position - node_c.position)
            })
        edge_df = Edge("DF", node_d, node_f, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": roof_angle,
            "length": np.linalg.norm(node_f.position - node_d.position)
            })
        edge_ef = Edge("EF", node_e, node_f, {
            "edge_type": EdgeType.EAVES,
            "slope_angle": roof_angle,
            "length": np.linalg.norm(node_f.position - node_e.position)
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

        self.edges = [edge_ac, edge_bc, edge_cf, edge_df, edge_ef, edge_ad, edge_be]

        # Generate faces
        face_acfd = Face(
            "ACFD",
            [node_a, node_c, node_f, node_d],
            [edge_ac, edge_cf, edge_df, edge_ad],
            {"face_type": "roof_plane"}  # TODO: Add face angle or tilt direction
        )

        face_bcfe = Face(
            "BCFE",
            [node_b, node_c, node_f, node_e],
            [edge_bc, edge_cf, edge_ef, edge_be],
            {"face_type": "roof_plane"}  # TODO: Add face angle or tilt direction
        )

        self.faces = [face_acfd, face_bcfe]

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

            print(f"tilt angle: {face.attributes['tilt_angle']}")
            print(f"tilt vector: {face.attributes['tilt_vector']}")
            print(f"centroid: {face.attributes['centroid']}")
            print(f"area: {face.attributes['area']}")

            face.attributes["roof_stype"] = "gable"
            face.attributes["roof_parameters"] = {
                "base_width": bw,
                "base_depth": bd,
                "ridge_height": rh
            }


if __name__ == "__main__":
    roof = GableRoof(10, 15, 7)

    # Get 3D structure
    structure_3d = roof.get_3d_structure()
    print("structure_3d")
    print(f"num of nodes: {len(structure_3d['nodes'])}")
    print(f"num of edges: {len(structure_3d['edges'])}")
    print(f"num of faces: {len(structure_3d['faces'])}")

    # Get node attrivutes
    print(structure_3d['nodes'][0]['attributes'])

    # Get edge attrivutes
    print(structure_3d['edges'][0]['attributes'])

    # Get face attrivutes
    print(structure_3d['faces'][0]['attributes'])

    # Get tilt angle from horizontal
    for face in structure_3d['faces']:
        print(f"Face id {face['id']}: tilt angle: {face['attributes']['tilt_angle']} degree")

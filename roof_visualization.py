import matplotlib.pyplot as plt
import numpy as np
import json
from enum import Enum
from models.gable_roof import GableRoof
from models.hip_roof import HipRoof
from models.base_roof import NodeType, EdgeType


# Custom JSON encoder for Enum
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Enumの値を返す
        # numpy配列への対応も追加
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def visualize_roof(roof):
    """ Visualize a roof object """
    # Get 3D structure
    structure = roof.get_3d_structure()

    # 3D plot
    fig = plt.figure(figsize=(15, 10))

    # 3D structure
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("3D Structure")

    # Draw nodes
    for node in structure["nodes"]:
        x, y, z = node["position"]

        # Change color based on node type
        node_type = node['attributes'].get('node_type')
        if node_type == NodeType.RIDGE.value:
            color = 'red'
        elif node_type == NodeType.EAVES.value:
            color = 'blue'
        else:
            color = 'gray'

        ax1.scatter(x, z, y, c=color, marker='o')
        ax1.text(x, z, y, node['id'])

    # Draw edges
    node_dict = {node['id']: node['position'] for node in structure['nodes']}
    for edge in structure['edges']:
        pos1 = node_dict[edge['start']]
        pos2 = node_dict[edge['end']]

        # Change color based on edge type
        edge_type = edge['attributes'].get('edge_type')
        if edge_type == EdgeType.EAVES.value:
            linestype = '-'
            linewidth = 1.5
            color = 'blue'
        elif edge_type == EdgeType.RIDGE.value:
            linestype = '-'
            linewidth = 2
            color = 'red'
        else:
            linestype = '--'
            linewidth = 1
            color = 'black'

        ax1.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]],
                 linestyle=linestype, linewidth=linewidth, color=color)

    # Draw faces
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'lightyellow', 'lightpink', 'lightcyan']
    for i, face in enumerate(structure['faces']):
        node_ids = face['node_ids']
        node_positions = [node_dict[node_id] for node_id in node_ids]
        x = [pos[0] for pos in node_positions]
        y = [pos[2] for pos in node_positions]
        z = [pos[1] for pos in node_positions]

        # Draw face
        alpha = 0.3

        # Change color based on face tilt angle
        face_color = colors[i % len(colors)]

        ax1.plot_trisurf(x, y, z, color=face_color, alpha=alpha)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_box_aspect([1, 1, 0.5])

    # Front View (XY plane)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Front View (XY plane)")

    front_view = roof.get_front_view()

    # Draw nodes
    for node in front_view['nodes']:
        x, y, = node['position']

        # Change color based on node type
        node_type = node['attributes'].get('node_type')
        if node_type == NodeType.RIDGE.value:
            color = 'red'
        elif node_type == NodeType.EAVES.value:
            color = 'blue'
        else:
            color = 'gray'

        ax2.scatter(x, y, c=color, marker='o')
        ax2.text(x, y, node['id'])

    # Draw edges
    node_dict_front = {node['id']: node['position'] for node in front_view['nodes']}
    for edge in front_view['edges']:
        if edge['start'] in node_dict_front and edge['end'] in node_dict_front:
            pos1 = node_dict_front[edge['start']]
            pos2 = node_dict_front[edge['end']]

            # Change color based on edge type
            edge_type = edge['attributes'].get('edge_type')
            if edge_type == EdgeType.EAVES.value:
                linestype = '-'
                linewidth = 1.5
                color = 'blue'
            elif edge_type == EdgeType.RIDGE.value:
                linestype = '-'
                linewidth = 2
                color = 'red'
            else:
                linestype = '--'
                linewidth = 1
                color = 'black'

            ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], linestyle=linestype, linewidth=linewidth, color=color)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')

    # Side Vide (ZY plane)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Side View (ZY plane)")

    side_view = roof.get_side_view()
    # Draw nodes
    for node in side_view['nodes']:
        z, y = node['position']

        node_type = node['attributes'].get('node_type')
        if node_type == NodeType.RIDGE.value:
            color = 'red'
        elif node_type == NodeType.EAVES.value:
            color = 'blue'
        else:
            color = 'gray'

        ax3.scatter(z, y, c=color, marker='o')
        ax3.text(z, y, node['id'])

    # Draw edges
    node_dict_side = {node['id']: node['position'] for node in side_view['nodes']}
    for edge in side_view['edges']:
        if edge['start'] in node_dict_side and edge['end'] in node_dict_side:
            pos1 = node_dict_side[edge['start']]
            pos2 = node_dict_side[edge['end']]

            edge_type = edge['attributes'].get('edge_type')
            if edge_type == EdgeType.EAVES.value:
                linestype = '-'
                linewidth = 1.5
                color = 'blue'
            elif edge_type == EdgeType.RIDGE.value:
                linestype = '-'
                linewidth = 2
                color = 'red'
            elif edge_type == EdgeType.VALLEY.value:
                linestype = '-'
                linewidth = 2
                color = 'purple'
            else:
                linestype = '--'
                linewidth = 1
                color = 'black'

            ax3.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], linestyle=linestype, linewidth=linewidth, color=color)

    ax3.set_xlabel('Z')
    ax3.set_ylabel('Y')
    ax3.grid(True)
    ax3.axis('equal')

    # Top View (ZX plane)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Top View (ZX plane)")

    top_view = roof.get_top_view()

    # Draw nodes
    for node in top_view['nodes']:
        x, z = node['position']

        node_type = node['attributes'].get('node_type')
        if node_type == NodeType.RIDGE.value:
            color = 'red'
        elif node_type == NodeType.EAVES.value:
            color = 'blue'
        else:
            color = 'gray'

        ax4.scatter(z, x, c=color, marker='o')
        ax4.text(z, x, node['id'])

    # Draw edges
    node_dict_top = {node['id']: node['position'] for node in top_view['nodes']}
    for edge in top_view['edges']:
        if edge['start'] in node_dict_top and edge['end'] in node_dict_top:
            pos1 = node_dict_top[edge['start']]
            pos2 = node_dict_top[edge['end']]

            edge_type = edge['attributes'].get('edge_type')
            if edge_type == EdgeType.EAVES.value:
                linestype = '-'
                linewidth = 1.5
                color = 'blue'
            elif edge_type == EdgeType.RIDGE.value:
                linestype = '-'
                linewidth = 2
                color = 'red'
            elif edge_type == EdgeType.VALLEY.value:
                linestype = '-'
                linewidth = 2
                color = 'purple'
            else:
                linestype = '--'
                linewidth = 1
                color = 'black'

            ax4.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]],
                     linestyle=linestype, linewidth=linewidth, color=color)

    ax4.set_xlabel('Z')
    ax4.set_ylabel('X')
    ax4.grid(True)
    ax4.axis('equal')

    plt.tight_layout()
    plt.show()


def export_roof_to_json(roof, filename):
    """ Export roof object to JSON file """
    data = {
        "3d_structure": roof.get_3d_structure(),
        "front_view": roof.get_front_view(),
        "side_view": roof.get_side_view(),
        "top_view": roof.get_top_view()
    }

    # Export to JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, cls=EnumEncoder)

    print(f"Roof data exported to {filename}")


def create_sample_dataset(roof_type, parameters_list, output_file):
    """ Create a sample dataset of roofs """
    dataset = []

    for i, params in enumerate(parameters_list):
        if roof_type == "gable":
            base_width, base_depth, ridge_height = params
            roof = GableRoof(base_width, base_depth, ridge_height)
        # TODO: Add more roof types here

        dataset.append({
            "id": f"{roof_type}_{i}",
            "parameters": params,
            "front_view": roof.get_front_view(),
            "side_view": roof.get_side_view(),
            "top_view": roof.get_top_view()
        })

    # JSON export
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2, cls=EnumEncoder)

    print(f"Sample dataset exported to {output_file}")


def analyze_roof_structure(roof):
    """ Analyze roof structure and display information """
    structure = roof.get_3d_structure()

    print("Statistics of roof:")
    print(f"Number of nodes: {len(structure['nodes'])}")
    print(f"Number of edges: {len(structure['edges'])}")
    print(f"Number of faces: {len(structure['faces'])}")

    # Node type distribution
    node_types = {}
    for node in structure['nodes']:
        node_type = node['attributes'].get('node_type')
        if node_type:
            node_types[node_type] = node_types.get(node_type, 0) + 1

    print("Node type distribution:")
    for type_name, count in node_types.items():
        print(f"{type_name}: {count}")

    # Edge type distribution
    edge_types = {}
    for edge in structure['edges']:
        edge_type = edge['attributes'].get('edge_type')
        if edge_type:
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print("Edge type distribution:")
    for type_name, count in edge_types.items():
        print(f"{type_name}: {count}")

    # Face type distribution
    print("Face type distribution:")
    for i, face in enumerate(structure['faces']):
        print(f"Face {i}: {face['attributes'].get('roof_type')}")
        print(f"Area: {face['attributes'].get('area', 'N/A'):.2f} m^2")
        print(f"Tilt angle: {face['attributes'].get('tilt_angle', 'N/A'):.2f} degree")
        print(f"Tilt vector: {face['attributes'].get('tilt_vector', 'N/A')}")
        print(f"Centroid: {face['attributes'].get('centroid', 'N/A')}")

    # Structure information
    print("Structure information:")
    all_positions = []
    for node in structure['nodes']:
        all_positions.append(node['position'])

    if all_positions:
        all_positions = np.array(all_positions)
        min_pos = np.min(all_positions, axis=0)
        max_pos = np.max(all_positions, axis=0)
        print(f"  X range: {min_pos[0]} - {max_pos[0]}")
        print(f"  Y range: {min_pos[1]} - {max_pos[1]}")
        print(f"  Z range: {min_pos[2]} - {max_pos[2]}")


def visualize_multiple_roofs(roofs, titles=None):
    """ Visualize multiple roofs in a single plot (only top view) """
    if titles is None:
        titles = [f"Roof {i}" for i in range(len(roofs))]

    fig, axes = plt.subplots(1, len(roofs), figsize=(5 * len(roofs), 5))
    if len(roofs) == 1:
        axes = [axes]

    for ax, roof, title in zip(axes, roofs, titles):
        top_view = roof.get_top_view()

        ax.set_title(title)

        # Draw nodes
        for node in top_view['nodes']:
            x, z = node['position']

            node_type = node['attributes'].get('node_type')
            if node_type == NodeType.RIDGE.value:
                color = 'red'
            elif node_type == NodeType.EAVES.value:
                color = 'blue'
            else:
                color = 'gray'

            ax.scatter(z, x, c=color, marker='o')
            ax.text(z, x, node['id'])

        # Draw edges
        node_dict = {node['id']: node['position'] for node in top_view['nodes']}
        for edge in top_view['edges']:
            if edge['start'] in node_dict and edge['end'] in node_dict:
                pos1 = node_dict[edge['start']]
                pos2 = node_dict[edge['end']]

                edge_type = edge['attributes'].get('edge_type')
                if edge_type == EdgeType.EAVES.value:
                    linestype = '-'
                    linewidth = 1.5
                    color = 'blue'
                elif edge_type == EdgeType.RIDGE.value:
                    linestype = '-'
                    linewidth = 2
                    color = 'red'
                elif edge_type == EdgeType.VALLEY.value:
                    linestype = '-'
                    linewidth = 2
                    color = 'purple'
                else:
                    linestype = '--'
                    linewidth = 1
                    color = 'black'

                ax.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]],
                        linestyle=linestype, linewidth=linewidth, color=color)

        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.grid(True)
        ax.axis('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gable_roof = GableRoof(10, 15, 7)

    # Visualize roof
    visualize_roof(gable_roof)

    hip_roof = HipRoof(10, 15, 7, 8)
    visualize_roof(hip_roof)

    # # Analyze roof structure
    # analyze_roof_structure(gable_roof)

    # # JSON export
    # export_roof_to_json(gable_roof, "gable_roof.json")

    # # Create sample dataset
    # gable_params = [
    #     [8.0, 6.0, 4.0],
    #     [10.0, 8.0, 5.0],
    #     [12.0, 10.0, 6.0]
    # ]
    # create_sample_dataset("gable", gable_params, "sample_roofs.json")

    # # Visualize multiple roofs
    # sample_roofs = [
    #     GableRoof(*params) for params in gable_params
    # ]
    # titles = [
    #     "Shallow Roof",
    #     "Medium Roof",
    #     "Steep Roof"
    # ]
    # visualize_multiple_roofs(sample_roofs, titles)


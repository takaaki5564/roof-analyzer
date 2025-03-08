import numpy as np
import json
import os
import sys
from enum import Enum
from tqdm import tqdm
import random


# Import roof model and classes
from roof_models.base_roof import BaseRoof, Node, Edge, Face, NodeType, EdgeType
from roof_models.gable_roof import GableRoof
from roof_models.hip_roof import HipRoof


# Custom JSON encoer for Enum
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_gable_roof_parameters(num_samples=100, seed=42):
    """ Generate parameters for GableRoof class """
    random.seed(seed)
    np.random.seed(seed)

    parameters = []

    base_widths = np.linspace(6, 15, 10)
    base_depths = np.linspace(6, 20, 10)
    ridge_heights = np.linspace(3, 10, 10)

    # Generate some systematic combinations
    for width in base_widths:
        for depth in base_depths[:3]:
            for height in ridge_heights[:3]:
                parameters.append([float(width), float(depth), float(height)])

    # Add some random combinations
    for _ in range(num_samples - len(parameters)):
        width = random.uniform(6, 15)
        depth = random.uniform(6, 20)
        height = random.uniform(3, 10)
        parameters.append([width, depth, height])

    # Shuffle and limit to num_samples
    random.shuffle(parameters)
    return parameters[:num_samples]


def generate_hip_roof_parameters(num_samples=100, seed=42):
    """ Generate parameters for HipRoof class """
    random.seed(seed)
    np.random.seed(seed)

    parameters = []

    base_widths = np.linspace(6, 15, 10)
    base_depths = np.linspace(6, 20, 10)
    ridge_heights = np.linspace(3, 10, 10)
    ridge_lengths = np.linspace(3, 10, 10)

    # Generate some systematic combinations
    for width in base_widths:
        for depth in base_depths[:3]:
            for height in ridge_heights[:3]:
                for length in ridge_lengths[:3]:
                    parameters.append([float(width), float(depth), float(height), float(length)])

    # Add some random combinations
    for _ in range(num_samples - len(parameters)):
        width = random.uniform(6, 15)
        depth = random.uniform(6, 20)
        height = random.uniform(3, 10)
        length = random.uniform(3, 10)
        parameters.append([width, depth, height, length])

    # Shuffle and limit to num_samples
    random.shuffle(parameters)
    return parameters[:num_samples]


def create_gmn_dataset(output_dir, num_samples=100):
    """ Create dataset for Graph Matching Network (GMN) training """
    os.makedirs(output_dir, exist_ok=True)

    # Generate parameters for gable roofs
    gable_params = generate_gable_roof_parameters(num_samples=num_samples)

    # Generate parameters for hip roofs
    hip_params = generate_hip_roof_parameters(num_samples=num_samples)

    dataset = []

    print(f"Generating {num_samples} samples for GableRoof and HipRoof models...")
    for i, params in tqdm(enumerate(gable_params + hip_params), total=num_samples * 2):
        if i < num_samples:
            # GableRoof
            base_width, base_depth, ridge_height = params

            roof = GableRoof(base_width, base_depth, ridge_height)

            front_view = roof.get_front_view()
            side_view = roof.get_side_view()
            top_view = roof.get_top_view()

            entry = {
                "id": f"gable_{i}",
                "parameters": {
                    "base_width": base_width,
                    "base_depth": base_depth,
                    "ridge_height": ridge_height
                },
                "input": {
                    "front_view": front_view,
                    "side_view": side_view
                },
                "target": {
                    "top_view": top_view
                }
            }

        else:
            # HipRoof
            base_width, base_depth, ridge_height, ridge_length = params

            roof = HipRoof(base_width, base_depth, ridge_height, ridge_length)

            front_view = roof.get_front_view()
            side_view = roof.get_side_view()
            top_view = roof.get_top_view()

            entry = {
                "id": f"hip_{i}",
                "parameters": {
                    "base_width": base_width,
                    "base_depth": base_depth,
                    "ridge_height": ridge_height,
                    "ridge_length": ridge_length
                },
                "input": {
                    "front_view": front_view,
                    "side_view": side_view
                },
                "target": {
                    "top_view": top_view
                }
            }

        dataset.append(entry)

    # Split into train and test
    random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]

    # Save as JSON files
    with open(os.path.join(output_dir, "train_dataset.json"), "w") as f:
        json.dump(train_set, f, cls=EnumEncoder, indent=2)

    with open(os.path.join(output_dir, "test_dataset.json"), "w") as f:
        json.dump(test_set, f, cls=EnumEncoder, indent=2)

    print(f"Dataset created: {len(train_set)} training samples and {len(test_set)} test samples")
    print(f"Saved to {output_dir}")

    return train_set, test_set


# Usage example
if __name__ == "__main__":
    output_dir = "data/gmn_dataset"
    create_gmn_dataset(output_dir, num_samples=200)

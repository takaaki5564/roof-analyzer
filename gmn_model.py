import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePssing, GCNConv
from torch_geometric.data import Data, Batch


class GraphEncoder(nn.Module):
    """ Graph encoder that embeds nodes based on their feature and graph structure"""

    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=32, num_layers=3):
        super(GraphEncoder, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node feature encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge feature encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr=None):
        # Encode node features
        x = self.node_encoder(x)

        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        return x


class GraphMatchingNetwork(nn.Module):
    """ Graph Matching Network that predicts top view based on fron and side views """

    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, output_node_dim=32, output_edge_dim=32):
        super(GraphMatchingNetwork, self).__init__()

        # Graph encoders for front and side views
        self.front_encoder = GraphEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        self.side_encoder = GraphEncoder(node_feature_dim, edge_feature_dim, hidden_dim)

        # Fusion layer to combine front and side view embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output layers for node and edge features
        self.node_output = nn.Linear(hidden_dim, output_node_dim)
        self.edge_output = nn.Linear(hidden_dim, output_edge_dim)

        # Node type classifier
        self.node_type_classifier = nn.Linear(hidden_dim, 2)  # Ridge, Eaves

        # Edge type classifier
        self.edge_type_classifier = nn.Linear(hidden_dim, 5)  # Ridge, Valley, Eaves, HipRidge, Gable

    def forward(self, front_data, side_data):
        # Encode front view graph
        front_node_emb = self.front_encoder(
            front_data.x,
            front_data.edge_index,
            front_data.edge_attr
        )

        # Encode side view graph
        side_node_emb = self.side_encoder(
            side_data.x,
            side_data.edge_index,
            side_data.edge_attr
        )

        # Fusion of embeddings (assuming node correspondence between views)
        # In practive, this requires more sophisticated alignment
        fused_emb = self.fusion_layer(torch.cat([front_node_emb, side_node_emb], dim=-1))

        # Generate node and edge features for top view
        node_features = self.node_output(fused_emb)

        # Predict node types (e.g. Ridge, Eaves)
        node_types = self.node_type_classifier(fused_emb)

        # For predicting edge types and connections, we would need more
        # complex logic that's beyond the scope of this exmample

        return {
            'node_features': node_features,
            'node_types': node_types
        }

    # Utility functions for converting between our roof data and Pytorch Geometric data
    def convert_view_to_torch_geometric(view_data):
        """ Convert view data to Pytorch Geometric Data object """

        # Extract node features - we'll use 3D positions and encode node type
        node_positions = []
        node_types = []
        node_id_to_idx = {}

        for i, node in enumerate(view_data['nodes']):
            # Store mapping from node ID to index
            node_id_to_idx[node['id']] = i

            # Extract position (2D for projections)
            position = node['position']
            node_positions.append(position)

            # Extract node type (one-hot encoded)
            node_type = 0
            if 'attributes' in node and 'node_type' in node['attributes']:
                if node['attributes']['node_type'] == 'ridge':
                    node_type = 1
                elif node['attributes']['node_type'] == 'eaves':
                    node_type = 0
            node_types.append(node_type)

        # Combine features
        nodes_tensor = torch.tensor(node_positions, dtype=torch.float)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)

        # Create one-hot encoding for node types
        node_types_one_hot = F.one_hot(node_types_tensor, num_classes=2).float()

        # Combine position and node type features
        node_features = torch.cat([nodes_tensor, node_types_one_hot], dim=1)

        # Extract edge features
        edge_indices = []
        edge_types = []

        for edge in view_data['edges']:
            # Get node indices
            start_idx = node_id_to_idx[edge['start']]
            end_idx = node_id_to_idx[edge['end']]

            # Add edge (both directions for undirected graph)
            edge_indices.append
            edge_indices.append([start_idx, end_idx])

            # Extract edge type
            edge_type = 0
            if 'attributes' in edge and 'edge_type' in edge['attributes']:
                if edge['attributes']['edge_type'] == 'ridge':
                    edge_type = 1
                elif edge['attributes']['edge_type'] == 'valley':
                    edge_type = 2
                elif edge['attributes']['edge_type'] == 'eaves':
                    edge_type = 3
                elif edge['attributes']['edge_type'] == 'hip_ridge':
                    edge_type = 4
                elif edge['attributes']['edge_type'] == 'gable':
                    edge_type = 0

            # Add edge type for both directions
            edge_types.append(edge_type)
            edge_types.append(edge_type)

        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)

        # Create one-hot encoding for edge types
        edge_types_one_hot = F.one_hot(edge_types_tensor, num_classes=5).float()

        # Create Pytorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_types_one_hot,
            num_nodes=len(view_data['nodes'])
        )

        return data, node_id_to_idx

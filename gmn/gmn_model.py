import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv
from torch_geometric.data import Batch
import numpy as np


def convert_view_to_torch_geometric(view_data):
    """
    Convert view data to PyTorch Geometric format

    Args:
        view_data: Dictionary containing nodes and edges

    Returns:
        data: PyTorch Geometric Data object
        node_map: Mapping from original node indices to new node indices
    """
    from torch_geometric.data import Data

    nodes = view_data["nodes"]
    edges = view_data["edges"]

    # Create node features
    num_nodes = len(nodes)
    node_features = []
    node_map = {}

    for i, node in enumerate(nodes):
        # Store mapping from original node ID to new index
        node_map[node["id"]] = i

        # Create feature vector: [x, y, one-hot node type]
        position = node["position"]
        
        # Get node type from attributes
        node_type = 0  # デフォルトはEAVES
        if "attributes" in node and "node_type" in node["attributes"]:
            node_type_str = node["attributes"]["node_type"].upper()
            if node_type_str == "RIDGE":
                node_type = 1
        
        # One-hot encode node type
        node_type_onehot = [0] * 2  # Assuming 2 node types
        node_type_onehot[node_type] = 1
        
        # Combine features
        features = [position[0], position[1]] + node_type_onehot
        node_features.append(features)

    # Create edge features and indices
    edge_indices = []
    edge_features = []

    for edge in edges:
        # エッジのsourceとtargetの取得方法を変更
        source_id = edge["source"] if "source" in edge else edge["start"]
        target_id = edge["target"] if "target" in edge else edge["end"]
        
        if source_id not in node_map or target_id not in node_map:
            print(f"Warning: Edge references unknown node: {source_id} -> {target_id}")
            continue
            
        source_idx = node_map[source_id]
        target_idx = node_map[target_id]
        
        # Get edge type from attributes
        edge_type = 0  # デフォルト
        if "attributes" in edge and "edge_type" in edge["attributes"]:
            edge_attr = edge["attributes"]["edge_type"].upper()
            if edge_attr == "RIDGE":
                edge_type = 0
            elif edge_attr == "VALLEY":
                edge_type = 1
            elif edge_attr == "EAVES":
                edge_type = 2
            elif edge_attr == "HIP_RIDGE":
                edge_type = 3
            elif edge_attr == "GABLE":
                edge_type = 4
        
        # One-hot encode edge type
        edge_type_onehot = [0] * 5  # 5種類のエッジタイプ
        edge_type_onehot[edge_type] = 1
        
        # Add edges in both directions
        edge_indices.append([source_idx, target_idx])
        edge_indices.append([target_idx, source_idx])
        
        # Add edge features for both directions
        edge_features.append(edge_type_onehot)
        edge_features.append(edge_type_onehot)

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)

    # エッジインデックスの範囲チェックを追加
    valid_edge_indices = []
    valid_edge_features = []
    for i, ((src, tgt), feat) in enumerate(zip(edge_indices, edge_features)):
        if src < len(nodes) and tgt < len(nodes):
            valid_edge_indices.append([src, tgt])
            valid_edge_features.append(feat)
        else:
            print(f"Warning: Edge {i} with indices ({src}, {tgt}) is out of range for {len(nodes)} nodes.")

    # エッジが存在する場合のみテンソルを作成
    if valid_edge_indices:
        edge_index = torch.tensor(valid_edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(valid_edge_features, dtype=torch.float)
    else:
        # エッジがない場合は空のテンソルを作成
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)  # 5種類のエッジタイプ

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, node_map


class GraphEncoder(nn.Module):
    """ Graph encoder that embeds nodes based on their feature and graph structure"""

    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, num_layers=3):
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

        # Edge feature encoding (optional)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(EdgeConv(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ), aggr='max'))  # 'max'集約を使用

    def forward(self, x, edge_index, edge_attr=None):
        # エッジインデックスの範囲チェック
        if edge_index.shape[1] > 0:  # エッジが存在する場合
            max_idx = edge_index.max().item()
            if max_idx >= x.size(0):
                print(f"Warning: Max edge index {max_idx} >= number of nodes {x.size(0)}")
                # 問題のあるエッジを除外
                valid_edges = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
                edge_index = edge_index[:, valid_edges]
                if edge_attr is not None:
                    edge_attr = edge_attr[valid_edges]

        # Encode node features
        x = self.node_encoder(x)

        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers):
            # エッジが存在する場合のみ畳み込みを適用
            if edge_index.shape[1] > 0:
                try:
                    x_new = conv(x, edge_index)
                    x = x_new
                except Exception as e:
                    print(f"Error in layer {i}: {e}")
                    # エラーが発生した場合は前の層の出力を維持
                    continue
            
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        return x


class EdgePredictor(nn.Module):
    """ Edge predictor that generates edge features based on node embeddings """
    def __init__(self, node_dim, edge_feature_dim, dropout=0.2):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
            # nn.Linear(node_dim, 1 + edge_feature_dim)  # Existance probability + edge type
        )

        self.existence_head = nn.Linear(node_dim, 1)
        self.type_head = nn.Linear(node_dim, edge_feature_dim)

    def forward(self, node_embeddings):
        num_nodes = node_embeddings.size(0)
        edge_predictions = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # ノード埋め込みの連結
                pair_embedding = torch.cat([node_embeddings[i], node_embeddings[j]], dim=0)
                
                # 共通エンコーディング
                encoded = self.edge_encoder(pair_embedding)
                
                # 存在確率とタイプを個別に予測
                existence_logit = self.existence_head(encoded)
                type_logits = self.type_head(encoded)
                
                edge_predictions.append((i, j, (existence_logit, type_logits)))

        return edge_predictions

class MLP(nn.Module):
    """多層パーセプトロン"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return self.layers[-1](x)


class GraphMatchingNetwork(nn.Module):
    """ Graph Matching Network that predicts top view based on front and side views """

    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, output_node_dim=4, output_edge_dim=5):
        super(GraphMatchingNetwork, self).__init__()

        # Graph encoders for front and side views
        self.front_encoder = GraphEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        self.side_encoder = GraphEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        
        # 学習可能なノードテンプレート（各ノードの相対的な位置関係を学習）
        self.register_parameter('node_templates', nn.Parameter(torch.randn(10, hidden_dim) * 0.02))
        
        # Global pooling and fusion layers
        self.front_pool = nn.Linear(hidden_dim, hidden_dim)
        self.side_pool = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layer to combine front and side view embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-attention for node feature generation
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 出力層
        self.node_pos_predictor = MLP(hidden_dim, hidden_dim, 2)  # XY座標予測
        self.node_type_classifier = MLP(hidden_dim, hidden_dim, 2)  # ノードタイプ分類

        # Predict edge features (optional)
        self.edge_predictor = EdgePredictor(hidden_dim, edge_feature_dim)

    def forward(self, front_view, side_view, top_view=None):
        """
        Forward pass of the Graph Matching Network

        Args:
            front_view: Front view graph data
            side_view: Side view graph data
            top_view: Top view graph data (学習時のみ使用)

        Returns:
            output: Dictionary with node features and node types
        """
        batch_size = 1  # 単一サンプルの場合は1

        # Encode front and side views
        front_node_emb = self.front_encoder(front_view.x, front_view.edge_index, front_view.edge_attr)
        side_node_emb = self.side_encoder(side_view.x, side_view.edge_index, side_view.edge_attr)
        
        # 修正: バッチのないグローバルプーリングを適切に処理
        front_batch = torch.zeros(front_node_emb.size(0), dtype=torch.long, device=front_node_emb.device)
        side_batch = torch.zeros(side_node_emb.size(0), dtype=torch.long, device=side_node_emb.device)
        
        front_global = global_mean_pool(front_node_emb, front_batch)  # [1, hidden_dim]
        side_global = global_mean_pool(side_node_emb, side_batch)     # [1, hidden_dim]
        
        # Project pooled embeddings
        front_global = self.front_pool(front_global)  # [1, hidden_dim]
        side_global = self.side_pool(side_global)     # [1, hidden_dim]
        
        # Concatenate global features
        global_features = torch.cat([front_global, side_global], dim=1)  # [1, hidden_dim*2]
        
        # Fuse global features
        fused_emb = self.fusion_layer(global_features)  # [1, hidden_dim]
        
        # 出力ノード数を決定
        if top_view is not None and self.training:
            num_output_nodes = top_view.x.size(0)
        else:
            num_output_nodes = 6  # デフォルト値
        
        # テンプレートを使って各ノードの初期埋め込みを生成（修正）
        if num_output_nodes <= self.node_templates.size(0):
            node_embeddings = self.node_templates[:num_output_nodes].clone()
        else:
            repeats = (num_output_nodes + self.node_templates.size(0) - 1) // self.node_templates.size(0)
            expanded_templates = self.node_templates.repeat(repeats, 1)
            node_embeddings = expanded_templates[:num_output_nodes]
        
        # クロスアテンション（修正）
        fused_emb_expanded = fused_emb.expand(num_output_nodes, -1)  # [num_nodes, hidden_dim]
        
        # 最終的なノード埋め込み（単純化）
        final_node_embeddings = node_embeddings + fused_emb_expanded
        
        # 出力層：ノード位置とタイプを予測
        node_positions = self.node_pos_predictor(final_node_embeddings)  # [num_nodes, 2]
        node_types = self.node_type_classifier(final_node_embeddings)    # [num_nodes, 2]
        
        edge_predictions = self.edge_predictor(final_node_embeddings)  # [num_edges, edge_feature_dim]
        # print(f"Generated {len(node_positions)} nodes and {len(edge_predictions)} edges.")

        return {
            'node_features': node_positions,
            'node_types': node_types,
            'edge_predictions': edge_predictions
        }
    
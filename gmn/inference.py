import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.base_roof import NodeType, EdgeType

# Import our model
from gmn.gmn_model import GraphMatchingNetwork, convert_view_to_torch_geometric


def load_model(model_path, config_path=None):
    """
    モデルをロードする関数
    
    Args:
        model_path: モデルファイルのパス
        config_path: 設定ファイルのパス（オプション）
        
    Returns:
        model: ロードされたモデル
        config: モデルの設定
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load config (either from separate file or from model checkpoint)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try to load config from model checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config if not found
            config = {
                'node_feature_dim': 4,
                'edge_feature_dim': 5,
                'hidden_dim': 128,
                'output_node_dim': 4,
                'output_edge_dim': 5
            }
    
    # Create model
    model = GraphMatchingNetwork(
        config['node_feature_dim'],
        config['edge_feature_dim'],
        config['hidden_dim'],
        config['output_node_dim'],
        config['output_edge_dim']
    )
    
    # Load model weights
    if os.path.basename(model_path).startswith("model_checkpoint") or os.path.basename(model_path) == "best_model.pt":
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Load just the model state
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model weights from {model_path}")
    
    # Set to evaluation mode
    model.eval()
    
    return model, config


def predict_top_view(model, front_view, side_view):
    """
    前面図と側面図から上面図を予測する関数
    
    Args:
        model: グラフマッチングネットワーク
        front_view: 前面図のデータ
        side_view: 側面図のデータ
        
    Returns:
        outputs: モデルの出力
        front_node_map: 前面図のノードマッピング
        side_node_map: 側面図のノードマッピング
    """
    # Convert views to PyTorch Geometric Data objects
    front_data, front_node_map = convert_view_to_torch_geometric(front_view)
    side_data, side_node_map = convert_view_to_torch_geometric(side_view)
    
    # Print input shapes for debugging
    print("Input shapes:")
    print(f"Front view: {front_data.x.size(0)} nodes, {front_data.edge_index.size(1)//2} edges")
    print(f"Side view: {side_data.x.size(0)} nodes, {side_data.edge_index.size(1)//2} edges")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(front_data, side_data)
    
    # Print output shape
    print("Output shape:")
    print(f"Node features: {outputs['node_features'].shape}")
    print(f"Node types: {outputs['node_types'].shape}")
    
    return outputs, front_node_map, side_node_map


def reconstruct_top_view_graph(prediction_outputs, front_view, side_view):
    """
    モデル出力から上面図のグラフを再構築する関数
    
    Args:
        prediction_outputs: モデルの出力
        front_view: 前面図のデータ
        side_view: 側面図のデータ
        
    Returns:
        top_view: 再構築された上面図
    """
    # Extract predicted node features and types
    node_positions = prediction_outputs['node_features'].cpu().numpy()
    node_types_logits = prediction_outputs['node_types'].cpu().numpy()
    node_types = np.argmax(node_types_logits, axis=1)
    
    # Create nodes for the top view
    nodes = []
    for i, (pos, node_type) in enumerate(zip(node_positions, node_types)):
        # Convert node type to string
        node_type_str = "RIDGE" if node_type == 1 else "EAVES"
        
        nodes.append({
            "id": f"N{i}",
            "position": pos.tolist(),
            "attributes": {
                "node_type": node_type_str
            }
        })
    
    # # Create edges connecting the nodes
    edges = []
    edge_id = 0
    
    # # Connect nodes that are likely to be connected based on proximity and node types
    # for i in range(len(nodes)):
    #     for j in range(i+1, len(nodes)):
    #         pos_i = np.array(nodes[i]["position"])
    #         pos_j = np.array(nodes[j]["position"])
            
    #         # Calculate distance between nodes
    #         dist = np.linalg.norm(pos_i - pos_j)
            
    #         # Threshold distance for connecting nodes - can be adjusted as needed
    #         threshold = 5.0
            
    #         # If nodes are close enough, add an edge
    #         if dist < threshold:
    #             # Determine edge type based on node types
    #             node_i_type = nodes[i]["attributes"]["node_type"]
    #             node_j_type = nodes[j]["attributes"]["node_type"]
                
    #             if node_i_type == "RIDGE" and node_j_type == "RIDGE":
    #                 edge_type = "RIDGE"
    #             elif node_i_type == "EAVES" and node_j_type == "EAVES":
    #                 edge_type = "EAVES"
    #             else:
    #                 # Mixed node types - assign based on spatial relationship
    #                 if abs(pos_i[1] - pos_j[1]) < abs(pos_i[0] - pos_j[0]):
    #                     # More horizontal than vertical - likely hip ridge
    #                     edge_type = "HIP_RIDGE"
    #                 else:
    #                     # More vertical - could be valley or gable
    #                     edge_type = "GABLE"
                
    #             edges.append({
    #                 "id": f"E{edge_id}",
    #                 "start": nodes[i]["id"],
    #                 "end": nodes[j]["id"],
    #                 "attributes": {
    #                     "edge_type": edge_type
    #                 }
    #             })
    #             edge_id += 1

    # エッジ予測のしきい値を調整（デフォルトよりも低く設定）
    existence_threshold = 0.3  # デフォルトの0.5よりも低く設定
    
    # 予測されたエッジを処理
    edge_predictions = prediction_outputs['edge_predictions']
    for i, j, (existence_logit, type_logits) in edge_predictions:
        # シグモイド関数で存在確率に変換
        existence_prob = torch.sigmoid(existence_logit).item()
        
        # しきい値よりも高い確率のエッジのみ追加
        if existence_prob > existence_threshold:
            # 最も確率の高いエッジタイプを選択
            edge_type_idx = torch.argmax(type_logits).item()
            edge_type_str = ["RIDGE", "VALLEY", "EAVES", "HIP_RIDGE", "GABLE"][edge_type_idx]
            
            # エッジの追加
            edges.append({
                "id": f"E{edge_id}",
                "start": nodes[i]["id"],
                "end": nodes[j]["id"],
                "attributes": {
                    "edge_type": edge_type_str,
                    "existence_prob": existence_prob  # デバッグ用に確率も保存
                }
            })
            edge_id += 1

    # Construct the top view dictionary
    top_view = {
        "projection_type": "top",
        "nodes": nodes,
        "edges": edges
    }
    
    return top_view


def visualize_prediction(original_top_view, predicted_top_view, save_path=None):
    """
    元の上面図と予測された上面図を可視化する関数
    
    Args:
        original_top_view: 元の上面図データ
        predicted_top_view: 予測された上面図データ
        save_path: 画像保存パス（オプション）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Common visualization parameters
    node_size = 100
    eaves_color = 'blue'
    ridge_color = 'red'
    edge_colors = {
        'RIDGE': 'red',
        'VALLEY': 'green',
        'EAVES': 'blue',
        'HIP_RIDGE': 'purple',
        'GABLE': 'orange'
    }
    
    # Plot original top view
    ax1.set_title("Original Top View", fontsize=14)
    
    # Plot nodes
    for node in original_top_view['nodes']:
        pos = node['position']
        node_type = node['attributes'].get('node_type', '').upper()
        
        color = ridge_color if node_type == 'RIDGE' else eaves_color
        ax1.scatter(pos[0], pos[1], c=color, s=node_size, edgecolors='black', zorder=3)
        ax1.annotate(node['id'], (pos[0], pos[1]), fontsize=10, 
                     ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    
    # Plot edges
    node_dict = {node['id']: node['position'] for node in original_top_view['nodes']}
    for edge in original_top_view['edges']:
        if edge['start'] in node_dict and edge['end'] in node_dict:
            pos1 = node_dict[edge['start']]
            pos2 = node_dict[edge['end']]
            
            edge_type = edge['attributes'].get('edge_type', '').upper()
            color = edge_colors.get(edge_type, 'black')
            
            # Line style based on edge type
            if edge_type == 'RIDGE':
                linestyle = '-'
                linewidth = 2.5
            elif edge_type == 'EAVES':
                linestyle = '-'
                linewidth = 2
            else:
                linestyle = '--'
                linewidth = 1.5
            
            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                     linestyle=linestyle, linewidth=linewidth, 
                     color=color, zorder=2)
            
            # Edge label - midpoint
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax1.annotate(edge['id'], (mid_x, mid_y), fontsize=8,
                         ha='center', va='center', backgroundcolor='white', zorder=4)
    
    # Plot predicted top view
    ax2.set_title("Predicted Top View", fontsize=14)
    
    # Plot nodes
    for node in predicted_top_view['nodes']:
        pos = node['position']
        node_type = node['attributes'].get('node_type', '').upper()
        
        color = ridge_color if node_type == 'RIDGE' else eaves_color
        ax2.scatter(pos[0], pos[1], c=color, s=node_size, edgecolors='black', zorder=3)
        ax2.annotate(node['id'], (pos[0], pos[1]), fontsize=10, 
                     ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    
    # Plot edges
    node_dict = {node['id']: node['position'] for node in predicted_top_view['nodes']}
    for edge in predicted_top_view['edges']:
        if edge['start'] in node_dict and edge['end'] in node_dict:
            pos1 = node_dict[edge['start']]
            pos2 = node_dict[edge['end']]
            
            edge_type = edge['attributes'].get('edge_type', '').upper()
            color = edge_colors.get(edge_type, 'black')
            
            # Line style based on edge type
            if edge_type == 'RIDGE':
                linestyle = '-'
                linewidth = 2.5
            elif edge_type == 'EAVES':
                linestyle = '-'
                linewidth = 2
            else:
                linestyle = '--'
                linewidth = 1.5
            
            ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                     linestyle=linestyle, linewidth=linewidth, 
                     color=color, zorder=2)
            
            # Edge label - midpoint
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax2.annotate(edge['id'], (mid_x, mid_y), fontsize=8,
                         ha='center', va='center', backgroundcolor='white', zorder=4)
    
    # Set axis labels and grid
    for ax in [ax1, ax2]:
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
    
    # Create legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    node_legends = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ridge_color, markersize=10, label='Ridge Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=eaves_color, markersize=10, label='Eaves Node')
    ]
    
    edge_legends = [
        Line2D([0], [0], color=edge_colors['RIDGE'], linewidth=2.5, linestyle='-', label='Ridge Edge'),
        Line2D([0], [0], color=edge_colors['EAVES'], linewidth=2, linestyle='-', label='Eaves Edge'),
        Line2D([0], [0], color=edge_colors['HIP_RIDGE'], linewidth=1.5, linestyle='--', label='Hip Ridge Edge'),
        Line2D([0], [0], color=edge_colors['GABLE'], linewidth=1.5, linestyle='--', label='Gable Edge')
    ]
    
    fig.legend(handles=node_legends + edge_legends, loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """メイン実行関数"""
    # 設定
    model_path = "models/gmn_model/best_model.pt"  # または "final_model.pt"
    config_path = "models/gmn_model/config.json"
    test_data_path = "data/test_dataset.json"
    output_dir = "predictions"
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルをロード
    print("Loading model...")
    try:
        model, config = load_model(model_path, config_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # テストデータをロード
    print(f"Loading test data from {test_data_path}")
    try:
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # テストサンプルのインデックス（必要に応じて変更）
    sample_indices = [0, 1, 2, 3, 4]  # 最初の5サンプル
    
    # 各サンプルに対して予測と可視化を実行
    for idx in sample_indices:
        if idx >= len(test_data):
            print(f"Sample index {idx} is out of range")
            continue
        
        print(f"\nProcessing test sample {idx}...")
        sample = test_data[idx]
        
        try:
            # 入力ビューを取得
            front_view = sample['input']['front_view']
            side_view = sample['input']['side_view']
            original_top_view = sample['target']['top_view']
            
            # 上面図を予測
            outputs, front_node_map, side_node_map = predict_top_view(model, front_view, side_view)
            
            # 予測結果から上面図グラフを再構築
            predicted_top_view = reconstruct_top_view_graph(outputs, front_view, side_view)
            
            # 予測結果を保存
            prediction_path = os.path.join(output_dir, f"prediction_{idx}.json")
            with open(prediction_path, 'w') as f:
                json.dump({
                    'original_top_view': original_top_view,
                    'predicted_top_view': predicted_top_view
                }, f, indent=2)
            print(f"Prediction saved to: {prediction_path}")
            
            # 結果を可視化
            vis_path = os.path.join(output_dir, f"visualization_{idx}.png")
            visualize_prediction(original_top_view, predicted_top_view, save_path=vis_path)
            
            # 予測結果の統計情報を表示
            print(f"Original top view: {len(original_top_view['nodes'])} nodes, {len(original_top_view['edges'])} edges")
            print(f"Predicted top view: {len(predicted_top_view['nodes'])} nodes, {len(predicted_top_view['edges'])} edges")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
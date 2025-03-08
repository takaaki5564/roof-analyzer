import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from gmn_model import GraphMatchingNetwork, convert_view_to_torch_geometric


class RoofGraphDataset(torch.utils.data.Dataset):
    """ Dataset class for roof graph data """

    def __init__(self, json_path):
        """
        Initialize the dataset from a JSON file

        Args:
            json_path: Path to JSON file containing dataset
        """
        self.data = []

        # Load data from JSON file
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            # Convert views to PyTorch Geometric data objects
            front_view, front_node_map = convert_view_to_torch_geometric(sample['input']['front_view'])
            side_view, side_node_map = convert_view_to_torch_geometric(sample['input']['side_view'])
            top_view, top_node_map = convert_view_to_torch_geometric(sample['target']['top_view'])

            top_view.top_view_data = sample['target']['top_view']
            top_view.node_map = top_node_map

            return {
                'id': sample['id'] if 'id' in sample else idx,
                'front_view': front_view,
                'side_view': side_view,
                'top_view': top_view,
                'front_node_map': front_node_map,
                'side_node_map': side_node_map,
                'top_node_map': top_node_map,
                'parameters': sample.get('parameters', {})
            }
        except Exception as e:
            print(f"Error processing sample 2 {idx}: {e}")
            # エラーが発生した場合はNoneを返す（DataLoaderでフィルタリング）
            return None


def collate_fn(batch):
    """DataLoader用のカスタムコレート関数 - Noneをフィルタリング"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch


# エッジ予測のための損失関数
def compute_edge_loss(predicted_edges, target_top_view, node_map_reverse):
    """
    エッジ予測の損失計算
    
    Args:
        predicted_edges: モデルによる予測エッジ [(i, j, (existence_logit, type_logits))]
        target_top_view: 目標上面図
        node_map_reverse: ノードIDからインデックスへのマッピング
    
    Returns:
        edge_loss: エッジ予測の総合損失
        edge_exist_loss: エッジ存在確率の損失
        edge_type_loss: エッジタイプ分類の損失
    """
   
    device = predicted_edges[0][2][0].device
    edge_exist_loss = 0
    edge_type_loss = 0
    
    # 目標エッジの準備
    target_edges = {}
    
    for edge in target_top_view['edges']:
        start_id = edge['start'] if 'start' in edge else edge['source']
        end_id = edge['end'] if 'end' in edge else edge['target']
        
        if start_id in node_map_reverse and end_id in node_map_reverse:
            i = node_map_reverse[start_id]
            j = node_map_reverse[end_id]
            
            # インデックスを小さい順に並べる
            i, j = min(i, j), max(i, j)
            
            # エッジタイプの取得
            edge_type = 0  # デフォルト
            if 'attributes' in edge and 'edge_type' in edge['attributes']:
                edge_type_str = edge['attributes']['edge_type'].upper()
                if edge_type_str == 'RIDGE':
                    edge_type = 0
                elif edge_type_str == 'VALLEY':
                    edge_type = 1
                elif edge_type_str == 'EAVES':
                    edge_type = 2
                elif edge_type_str == 'HIP_RIDGE':
                    edge_type = 3
                elif edge_type_str == 'GABLE':
                    edge_type = 4
            
            target_edges[(i, j)] = edge_type
    # print("DEBUG3-001", predicted_edges)
    # 各予測エッジと目標を比較
    for i, j, edge_feature in predicted_edges:
        # print("edge_data: ", edge_data)
        existence_logit = edge_feature[0]
        type_logits = edge_feature[1:]
        # print(f"Debug - existence_logit: {existence_logit.shape}, value: {existence_logit}")
    
        # エッジの存在確率の損失 - テンソルのサイズを合わせる
        # target_exists = torch.tensor(1.0 if (i, j) in target_edges else 0.0, device=device)
        # # スカラーを使用して計算
        # edge_exist_loss += F.binary_cross_entropy_with_logits(existence_logit, target_exists)

        # 修正版 - ターゲットを1次元テンソルに変更
        target_exists = torch.tensor([1.0 if (i, j) in target_edges else 0.0], device=device)
        # print(f"Debug - target_exists: {target_exists.shape}, value: {target_exists}")
    
        # unsqueezeを使わない - すでに両方が[1]サイズ
        edge_exist_loss += F.binary_cross_entropy_with_logits(existence_logit, target_exists)

        # エッジが実際に存在する場合、タイプの損失も計算
        if (i, j) in target_edges:
            edge_type = target_edges[(i, j)]
            target_type = torch.tensor(edge_type, device=device, dtype=torch.long)
            # print(f"Debug - type_logits: {type(type_logits)}, shape: {type_logits.shape if hasattr(type_logits, 'shape') else 'no shape'}, value: {type_logits}")
            # print(f"Debug - target_type: {type(target_type)}, shape: {target_type.shape}, value: {target_type}")
            # もしtype_logitsがタプルなら、最初の要素を使用
            if isinstance(type_logits, tuple):
                type_logits = type_logits[0]

            edge_type_loss += F.cross_entropy(type_logits.unsqueeze(0), target_type.unsqueeze(0))

    # 損失の重み付け - エッジタイプよりも存在確率を重視
    edge_exist_loss = edge_exist_loss / max(1, len(predicted_edges))
    edge_type_loss = edge_type_loss / max(1, len([e for e in predicted_edges if (e[0], e[1]) in target_edges]))

    # 総合損失
    total_loss = 3.0 * edge_exist_loss + edge_type_loss

    return total_loss, edge_exist_loss.item(), edge_type_loss.item()


def process_single_sample(model, front_view, side_view, top_view, criterion_mse, criterion_ce, optimizer=None, train=True):
    """
    単一のサンプルを処理する関数
    
    Args:
        model: モデル
        front_view: 前面図
        side_view: 側面図
        top_view: 上面図
        criterion_mse: MSE損失関数（座標用）
        criterion_ce: クロスエントロピー損失関数（ノードタイプ用）
        optimizer: オプティマイザ（訓練時のみ使用）
        train: 訓練モードかどうか
        
    Returns:
        loss: 損失値
    """
    # Target data extraction
    target_node_positions = top_view.x[:, :2]  # 2D positions
    target_node_types = torch.argmax(top_view.x[:, 2:], dim=1)  # Node types

    # ノードマッピングの構築（エッジ損失計算用）
    top_view_data = None
    node_map_reverse = {}
    
    if hasattr(top_view, 'top_view_data'):
        top_view_data = top_view.top_view_data
        
        # ノードIDからインデックスへのマッピング作成
        if hasattr(top_view, 'node_map'):
            node_map = top_view.node_map
            for node_id, idx in node_map.items():
                node_map_reverse[node_id] = idx
    else:
        print("Warning: No top view data")
    
    if hasattr(top_view, 'node_map'):
        node_map = top_view.node_map
        for node_id, idx in node_map.items():
            node_map_reverse[node_id] = idx
        # print(f"Created node map reverse with {len(node_map_reverse)} entries")
    else:
        print("Warning: No node map")

    if train and optimizer is not None:
        optimizer.zero_grad()
    
    try:
        # Forward pass
        outputs = model(front_view, side_view, top_view if train else None)
        
        # Get predictions
        predicted_node_positions = outputs['node_features']
        predicted_node_types = outputs['node_types']
        edge_predictions = outputs['edge_predictions']
        
        # Handle size mismatch
        if predicted_node_positions.size(0) != target_node_positions.size(0):
            if train:
                print(f"Size mismatch - Pred: {predicted_node_positions.size(0)}, Target: {target_node_positions.size(0)}")
            
            # Adjust sizes for loss calculation
            if predicted_node_positions.size(0) < target_node_positions.size(0):
                # Pad predictions
                padding_size = target_node_positions.size(0) - predicted_node_positions.size(0)
                pos_padding = torch.zeros(padding_size, 2, device=predicted_node_positions.device)
                type_padding = torch.zeros(padding_size, 2, device=predicted_node_types.device)
                
                predicted_node_positions = torch.cat([predicted_node_positions, pos_padding], dim=0)
                predicted_node_types = torch.cat([predicted_node_types, type_padding], dim=0)
                
                # position_loss = criterion_mse(padded_positions, target_node_positions)
                # type_loss = criterion_ce(padded_types, target_node_types)
            else:
                # Truncate predictions
                # 出力が大きい場合は切り詰め
                predicted_node_positions = predicted_node_positions[:target_node_positions.size(0)]
                predicted_node_types = predicted_node_types[:target_node_types.size(0)]        
                # position_loss = criterion_mse(predicted_node_positions[:target_node_positions.size(0)], target_node_positions)
                # type_loss = criterion_ce(predicted_node_types[:target_node_types.size(0)], target_node_types)
        # else:
        #     # Sizes match
        #     position_loss = criterion_mse(predicted_node_positions, target_node_positions)
        #     type_loss = criterion_ce(predicted_node_types, target_node_types)

        # ノード位置の損失
        position_loss = criterion_mse(predicted_node_positions, target_node_positions)
        # ノードタイプの損失
        type_loss = criterion_ce(predicted_node_types, target_node_types)

        # エッジ予測の損失（追加）
        edge_loss = torch.tensor(0.0, device=predicted_node_positions.device)
        edge_exist_loss = 0.0
        edge_type_loss = 0.0        

        if edge_predictions and top_view_data and node_map_reverse:
            # loss_debug = compute_edge_loss(
            #     edge_predictions, top_view_data, node_map_reverse
            # )
            # print(f"Edge loss debug: {loss_debug}")
            edge_loss, edge_exist_loss, edge_type_loss = compute_edge_loss(
                edge_predictions, top_view_data, node_map_reverse
            )
        # Total loss - 位置とタイプの損失を重み付け
        # loss = position_loss + 0.5 * type_loss
        total_loss = position_loss + 0.5 * type_loss + 2.0 * edge_loss
                
        if train and optimizer is not None:
            total_loss.backward()
            optimizer.step()
        
        return (
            total_loss.item(),
            position_loss.item(),
            type_loss.item(),
            edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
            edge_exist_loss,
            edge_type_loss
        )
    
    except Exception as e:
        print(f"Error processing sample 1: {str(e)}")
        if hasattr(front_view, 'x') and hasattr(side_view, 'x') and hasattr(top_view, 'x'):
            print(f"Nodes - Front: {front_view.x.size(0)}, Side: {side_view.x.size(0)}, Top: {top_view.x.size(0)}")
        return (None, None, None, None, None, None)


def train_gmn_model(train_data_path, test_data_path=None, output_dir="models/gmn_model", config=None):
    """ Train Graph Matching Network """

    if config is None:
        config = {
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'node_feature_dim': 4,   # 2D position + one-hot node type (2 classes)
            'edge_feature_dim': 5,   # one-hot edge type (5 classes)
            'output_node_dim': 4,    # 2D position + one-hot node type (2 classes)
            'output_edge_dim': 5,    # one-hot edge type (5 classes)
            'weight_decay': 1e-5,    # L2正則化のパラメータ
            'early_stopping': 15,    # 何エポック改善がなければ早期終了するか
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create dataset
    full_dataset = RoofGraphDataset(train_data_path)
    
    # Test datasetがない場合は、trainを分割する
    if test_data_path is None:
        train_size = int(len(full_dataset) * 0.8)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        val_dataset = RoofGraphDataset(test_data_path)
    
    # サンプル数を表示
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Check node distributions (for debugging)
    front_nodes = []
    side_nodes = []
    top_nodes = []
    
    for i in range(min(10, len(full_dataset))):
        sample = full_dataset[i]
        if sample is not None:
            front_nodes.append(sample['front_view'].x.size(0))
            side_nodes.append(sample['side_view'].x.size(0))
            top_nodes.append(sample['top_view'].x.size(0))
    
    print(f"Front view nodes: {front_nodes}")
    print(f"Side view nodes: {side_nodes}")
    print(f"Top view nodes: {top_nodes}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = GraphMatchingNetwork(
        config['node_feature_dim'],
        config['edge_feature_dim'],
        config['hidden_dim'],
        config['output_node_dim'],
        config['output_edge_dim']
    ).to(device)

    # Define loss and optimizer
    criterion_mse = nn.MSELoss()  # 座標予測用
    criterion_ce = nn.CrossEntropyLoss()  # ノードタイプ分類用
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # L2正則化
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )

    # Initialize tracking variables
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    
    # Losses storage
    train_losses = []
    val_losses = []
    position_losses = []
    type_losses = []

    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pos_loss = 0.0
        train_type_loss = 0.0
        train_edge_loss = 0.0
        train_edge_exist_loss = 0.0
        train_edge_type_loss = 0.0
        valid_batches = 0
        valid_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        for i, batch in enumerate(progress_bar):
            if batch is None:
                continue
                
            batch_loss = 0.0
            batch_pos_loss = 0.0
            batch_type_loss = 0.0
            batch_valid_samples = 0
            batch_edge_loss = 0.0
            batch_edge_exist_loss = 0.0
            batch_edge_type_loss = 0.0
            
            # Process each sample in the batch
            for sample in batch:
                # Move data to device
                front_view = sample['front_view'].to(device)
                side_view = sample['side_view'].to(device)
                top_view = sample['top_view'].to(device)
                
                # Process the sample
                sample_results = process_single_sample(
                    model, front_view, side_view, top_view, 
                    criterion_mse, criterion_ce, optimizer, train=True
                )
                
                if sample_results[0] is not None:
                    sample_loss, pos_loss, type_loss, edge_loss, edge_exist_loss, edge_type_loss = sample_results
                    batch_loss += sample_loss
                    batch_pos_loss += pos_loss
                    batch_type_loss += type_loss
                    batch_edge_loss += edge_loss
                    batch_edge_exist_loss += edge_exist_loss
                    batch_edge_type_loss += edge_type_loss

                    batch_valid_samples += 1
            
            # Update batch statistics
            if batch_valid_samples > 0:
                avg_batch_loss = batch_loss / batch_valid_samples
                avg_batch_pos_loss = batch_pos_loss / batch_valid_samples
                avg_batch_type_loss = batch_type_loss / batch_valid_samples
                avg_batch_edge_loss = batch_edge_loss / batch_valid_samples
                avg_batch_edge_exist_loss = batch_edge_exist_loss / batch_valid_samples
                avg_batch_edge_type_loss = batch_edge_type_loss / batch_valid_samples
                
                train_loss += avg_batch_loss
                train_pos_loss += avg_batch_pos_loss
                train_type_loss += avg_batch_type_loss
                train_edge_loss += avg_batch_edge_loss
                train_edge_exist_loss += avg_batch_edge_exist_loss
                train_edge_type_loss += avg_batch_edge_type_loss

                valid_batches += 1
                valid_samples += batch_valid_samples
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{avg_batch_loss:.4f}",
                    'pos_loss': f"{avg_batch_pos_loss:.4f}",
                    'type_loss': f"{avg_batch_type_loss:.4f}",
                    'valid': batch_valid_samples,
                    'edge_loss': f"{avg_batch_edge_loss:.4f}",
                    'edge_exist_loss': f"{avg_batch_edge_exist_loss:.4f}",
                    'edge_type_loss': f"{avg_batch_edge_type_loss:.4f}"
                })
        
        # Calculate epoch statistics
        if valid_batches > 0:
            epoch_train_loss = train_loss / valid_batches
            epoch_pos_loss = train_pos_loss / valid_batches
            epoch_type_loss = train_type_loss / valid_batches
        else:
            epoch_train_loss = float('inf')
            epoch_pos_loss = float('inf')
            epoch_type_loss = float('inf')
            print("Warning: No valid batches in this epoch")
        
        train_losses.append(epoch_train_loss)
        position_losses.append(epoch_pos_loss)
        type_losses.append(epoch_type_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_valid_batches = 0
        val_valid_samples = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Validation {epoch + 1}")
            for batch in val_progress_bar:
                if batch is None:
                    continue
                    
                batch_loss = 0.0
                batch_valid_samples = 0
                
                # Process each sample in the batch
                for sample in batch:
                    # Move data to device
                    front_view = sample['front_view'].to(device)
                    side_view = sample['side_view'].to(device)
                    top_view = sample['top_view'].to(device)
                    
                    # Process the sample
                    sample_results = process_single_sample(
                        model, front_view, side_view, top_view, 
                        criterion_mse, criterion_ce, None, train=False
                    )
                    
                    if sample_results[0] is not None:
                        # sample_loss, _, _, _, _, _ = sample_results
                        sample_loss = sample_results[0]
                        batch_loss += sample_loss
                        batch_valid_samples += 1
                
                # Update batch statistics
                if batch_valid_samples > 0:
                    avg_batch_loss = batch_loss / batch_valid_samples
                    val_loss += avg_batch_loss
                    val_valid_batches += 1
                    val_valid_samples += batch_valid_samples
                    
                    # Update progress bar
                    val_progress_bar.set_postfix({
                        'val_loss': f"{avg_batch_loss:.4f}",
                        'valid': batch_valid_samples
                    })
        
        # Calculate validation epoch statistics
        if val_valid_batches > 0:
            epoch_val_loss = val_loss / val_valid_batches
        else:
            epoch_val_loss = float('inf')
            print("Warning: No valid validation batches in this epoch")
        
        val_losses.append(epoch_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, "
              f"Train Loss: {epoch_train_loss:.4f} (Pos: {epoch_pos_loss:.4f}, Type: {epoch_type_loss:.4f}), "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Valid Samples: {valid_samples}/{len(train_dataset)}")
        
        # Check for best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'config': config
            }, os.path.join(output_dir, "best_model.pt"))
            
            print(f"New best model saved (validation loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss
            }, os.path.join(output_dir, f"model_checkpoint_epoch{epoch + 1}.pt"))
        
        # Early stopping check
        if early_stop_counter >= config['early_stopping']:
            print(f"Early stopping triggered after {epoch + 1} epochs. "
                  f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    
    # Save losses to file
    loss_data = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'position_loss': position_losses,
        'type_loss': type_losses
    }
    np.savez(os.path.join(output_dir, "training_history.npz"), **loss_data)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, position_losses, type_losses, output_dir)
    
    print(f"Training complete. Best model saved at epoch {best_epoch + 1} "
          f"with validation loss: {best_val_loss:.4f}")
    
    return model


def plot_training_curves(train_losses, val_losses, position_losses, type_losses, output_dir):
    """訓練曲線をプロットする関数"""
    # Filter out infinity values
    epochs = range(1, len(train_losses) + 1)
    valid_train = [l for l in train_losses if not math.isinf(l)]
    valid_val = [l for l in val_losses if not math.isinf(l)]
    valid_pos = [l for l in position_losses if not math.isinf(l)]
    valid_type = [l for l in type_losses if not math.isinf(l)]
    valid_epochs = range(1, len(valid_train) + 1)
    
    # Total loss plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if valid_train:
        plt.plot(valid_epochs, valid_train, label='Training Loss')
    if valid_val:
        plt.plot(valid_epochs, valid_val, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Component losses plot
    plt.subplot(1, 2, 2)
    if valid_pos:
        plt.plot(valid_epochs, valid_pos, label='Position Loss')
    if valid_type:
        plt.plot(valid_epochs, valid_type, label='Type Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Component Loss')
    plt.title('Position and Type Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
    
    # Learning curve (log scale)
    plt.figure(figsize=(10, 6))
    if valid_train:
        plt.semilogy(valid_epochs, valid_train, label='Training Loss')
    if valid_val:
        plt.semilogy(valid_epochs, valid_val, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_curve_log.png"))
    plt.close()


if __name__ == "__main__":
    train_data_path = "dataset/train_dataset.json"
    test_data_path = "dataset/test_dataset.json"
    output_dir = "gmn/gmn_model"
    
    # モデル構成のカスタマイズ（必要に応じて）
    custom_config = {
        'batch_size': 8,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'node_feature_dim': 4,  # 2D position + one-hot node type (2 classes)
        'edge_feature_dim': 5,  # one-hot edge type (5 classes)
        'output_node_dim': 4,   # 2D position + one-hot node type (2 classes)
        'output_edge_dim': 5,   # one-hot edge type (5 classes)
        'weight_decay': 1e-5,   # L2正則化のパラメータ
        'early_stopping': 20,   # 何エポック改善がなければ早期終了するか
    }

    # Train the model
    model = train_gmn_model(train_data_path, test_data_path, output_dir, custom_config)
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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

        # Load data from JSOn file
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def _getitem__(self, idx):
        sample = self.data[idx]

        # Convert views to PyTorch Geometric data objects
        front_view, front_node_map = convert_view_to_torch_geometric(sample['input']['front_view'])
        side_view, side_node_map = convert_view_to_torch_geometric(sample['input']['side_view'])
        top_view, top_node_map = convert_view_to_torch_geometric(sample['target']['top_view'])

        return {
            'id': sample['id'],
            'front_view': front_view,
            'side_view': side_view,
            'top_view': top_view,
            'front_node_map': front_node_map,
            'side_node_map': side_node_map,
            'top_node_map': top_node_map,
            'parameters': sample['parameters']
        }


def train_gmn_model(train_data_path, test_data_path, output_dir, config=None):
    """ Train Graph Matching Network """

    if config is None:
        config = {
            'batch_size': 16,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'hidden_dim': 64,
            'node_feature_dim': 4,  # 2D position + one-hot node type
            'edge_feature_dim': 3,  # one-hot edge type
            'output_node_dim': 4,
            'output_edge_dim': 3
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create datasets
    train_dataset = RoofGraphDataset(train_data_path)
    test_dataset = RoofGraphDataset(test_data_path)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphMatchingNetwork(
        config['node_feature_dim'],
        config['edge_feature_dim'],
        config['hidden_dim'],
        config['output_node_dim'],
        config['output_edge_dim']
    ).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    train_losses = []
    test_losses = []

    print(f"Training on {device}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        for batch in progress_bar:
            # Move data to device
            front_view = batch['front_view'].to(device)
            side_view = batch['side_view'].to(device)
            top_view = batch['top_view'].to(device)

            # Extract target node types
            target_node_features = top_view.x[:, :2]  # 2D positions
            target_node_types = torch.argmax(top_view.x[:, 2:], dim=1)  # Node types

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(front_view, side_view)

            # Compute loss
            feature_loss = criterion(outputs['node_features'], target_node_features)
            type_loss = classifier_criterion(outputs['node_types'], target_node_types)

            # Total loss
            loss = feature_loss + type_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (progress_bar.n + 1)})

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                front_view = batch['front_view'].to(device)
                side_view = batch['side_view'].to(device)
                top_view = batch['top_view'].to(device)

                # Extract target node types
                target_node_features = top_view.x[:, :2]  # 2D positions
                target_node_types = torch.argmax(top_view.x[:, 2:], dim=1)  # Node types

                # Forward pass
                outputs = model(front_view, side_view)

                # Compute loss
                feature_loss = criterion(outputs['node_features'], target_node_features)
                type_loss = classifier_criterion(outputs['node_types'], target_node_types)

                # Total loss
                loss = feature_loss + type_loss
                test_loss += loss.item()

        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss
            }, os.path.join(output_dir, f"model_checkpoint_epoch{epoch + 1}.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))

    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))

    # Save loss values
    np.savez(
        os.path.join(output_dir, "training_history.npz"),
        train=np.array(train_losses),
        test=np.array(test_losses)
    )

    print(f"Training complete. Model saved to {output_dir}")

    return model


if __name__ == "__main__":
    train_data_path = "data/train_dataset.json"
    test_data_path = "data/test_dataset.json"
    output_dir = "models/gmn_model"

    # Train the model
    model = train_gmn_model(train_data_path, test_data_path, output_dir)




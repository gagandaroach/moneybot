import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset.dataset import StocksDataset
from models.lstm import StockLSTM

def train_model():
    # Hyperparameters
    sequence_length = 60
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # Initialize datasets and loaders
    train_dataset = StocksDataset(sequence_length=sequence_length, train=True)
    val_dataset = StocksDataset(sequence_length=sequence_length, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss, optimizer
    model = StockLSTM(input_size=20, hidden_size=50, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets.squeeze(-1))  # Remove last dim if needed
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, targets.squeeze(-1))
                val_loss += loss.item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_stock_lstm.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f'Training complete. Best validation loss: {best_val_loss:.6f}')

if __name__ == '__main__':
    train_model()
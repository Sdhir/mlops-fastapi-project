# Model Training with PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
from sklearn.metrics import precision_score, recall_score

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        # FIXME: Implement the model architecture
        # Hint: Consider adding hidden layers and dropout
        self.fc1 = nn.Linear(input_dim, 128)
        # Add hidden layers
        self.hidden2 = nn.Linear(128, 64)
        # Add activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # Add a final layer to map to the output dimension
        self.output_layer = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # FIXME: Implement the forward pass
        # Hint: Use the defined layers and activation functions
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        out = self.softmax(self.output_layer(x))
        return out

# Visualize sample predictions
def log_predictions(model, data, target, num_samples=10):
    # FIXME: Implement this function to log prediction samples
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        # Log the predictions
        for i in range(num_samples):
            wandb.log({
                "input_images": wandb.Image(data[i], caption=f"GT: {target[i].item()}, Pred: {pred[i].item()}")
            })

def train_model():

    # Hyperparameters to experiment with
    config = {
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 64,
        "hidden_layers": [128, 64],  # Example: two hidden layers
        "dropout_rate": 0.2
    }

    # Initialize wandb
    wandb.init(project="mnist-mlops", config=config)

    # Load and preprocess data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model, loss, and optimizer
    model = LogisticRegression(input_dim=28*28, output_dim=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Add this after model initialization
    wandb.watch(model, log="all")

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # FIXME: Implement the training step
            # Hint: Remember to log the loss with wandb
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # Log the loss
            wandb.log({"loss": loss.item()})
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

        # In the training loop, add more detailed logging
        # FIXME: Log additional metrics (e.g., learning rate, gradient norms)
        # Log the learning rate
        for param_group in optimizer.param_groups:
            wandb.log({"learning_rate": param_group['lr']})
        # Log the gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({f"grad_norm_{name}": param.grad.norm()})

        # Log sample predictions
        # Call this function at the end of each epoch or at the end of training
        log_predictions(model, next(iter(test_loader))[0][:10], next(iter(test_loader))[1][:10])

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            # FIXME: Implement the validation step
            # Hint: Remember to log the accuracy with wandb
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nValidation set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
        # Log the accuracy
        wandb.log({"accuracy": accuracy})
        # After validation, log more metrics
        # FIXME: Log additional validation metrics (e.g., precision, recall)
        # Log precision and recall
        precision = precision_score(target, pred, average='weighted')
        recall = recall_score(target, pred, average='weighted')
        wandb.log({"precision": precision, "recall": recall})
    # Save the last model checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    wandb.finish()

if __name__ == "__main__":
    train_model()

    
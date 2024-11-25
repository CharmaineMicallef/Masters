import os
import csv
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision import datasets 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to dataset directory
data_dir = r'dataset'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load entire dataset
train_folder = os.path.join(data_dir, 'train')
val_folder = os.path.join(data_dir, 'val')

# Merge val/ back into train/ for dynamic splitting
full_dataset = ImageFolder(root=train_folder, transform=transform)

# Get class-wise indices
labels = full_dataset.targets
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,  # Adjust validation size (e.g., 20%)
    random_state=42,
    stratify=labels  # Maintain class balance
)

# Create training and validation subsets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# define CNN model
class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 80 * 80, 256)  # Adjust input dimensions
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 80 * 80)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    








# # initialising CNN model, loss function and optimiser
# model = FireCNN().to(device) # move model to specified device
# # Count occurrences of each class in the training set
# # Calculate class weights from training dataset
# train_labels = [label for _, label in train_dataset]
# class_counts = Counter(train_labels)
# pos_weight = class_counts[0] / class_counts[1]

# class_weights = torch.tensor([pos_weight], dtype=torch.float).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# # Print class weights for verification
# print(f"Class weights: {class_weights}")

# # Optimizer with weight decay
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# # Cyclic learning rate scheduler (alternative to ReduceLROnPlateau)
# scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2'
# )

# # Create a weighted sampler for class imbalance handling
# weights = [1.0 / class_counts[label] for label in train_labels]
# sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# # DataLoader with weighted sampling
# train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# # Open CSV file for logging metrics
# metrics_file = open(r'model_metrics_16_.csv', mode='w', newline='')
# metrics_writer = csv.writer(metrics_file)
# # Write header
# metrics_writer.writerow([
#     'Epoch',
#     'Training Accuracy', 'Training Loss', 'Training Precision', 'Training Recall', 'Training F1 Score',
#     'Validation Accuracy', 'Validation Loss', 'Validation Precision', 'Validation Recall', 'Validation F1 Score'
# ])


# # Lists to store metrics for plotting
# train_accuracies = []
# train_losses = []
# val_accuracies = []
# val_losses = []
# val_precisions = []
# val_recalls = []
# val_f1_scores = []


# # function to train the model
# def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10, patience=5):
#     best_val_loss = float('inf')
#     patience_counter = 0

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         all_train_preds = []
#         all_train_labels = []

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.float().to(device)
#             optimizer.zero_grad()

#             outputs = model(inputs).view(-1)  # Ensure 1D outputs
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#             preds = (outputs > 0.5).float()  # Thresholding
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#             all_train_preds.extend(preds.cpu().numpy())
#             all_train_labels.extend(labels.cpu().numpy())

#         average_train_loss = running_loss / len(train_loader)
#         training_accuracy = correct / total

#         train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
#         train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
#         train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         print(f"  Training - Loss: {average_train_loss:.4f}, Accuracy: {training_accuracy:.4f}, "
#               f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}")

#         # Pass criterion to validate_model
#         val_accuracy, val_loss, val_precision, val_recall, val_f1 = validate_model(model, val_loader, criterion)
#         scheduler.step(val_loss)

#         print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f},"
#               f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")

#         # Log metrics with clear differentiation
#         metrics_writer.writerow([
#             epoch + 1,
#             training_accuracy, average_train_loss, train_precision, train_recall, train_f1,
#             val_accuracy, val_loss, val_precision, val_recall, val_f1
#         ])

#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break

#         # Store metrics for plotting
#         train_accuracies.append(training_accuracy)
#         train_losses.append(average_train_loss)
#         val_accuracies.append(val_accuracy)
#         val_losses.append(val_loss)
#         val_precisions.append(val_precision)
#         val_recalls.append(val_recall)
#         val_f1_scores.append(val_f1)


        
# # function to validate model
# def validate_model(model, val_loader, criterion):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     running_val_loss = 0.0

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.float().to(device)
#             outputs = model(inputs).view(-1)
#             preds = (outputs > 0.5).float()

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#             val_loss = criterion(outputs, labels)
#             running_val_loss += val_loss.item()

#     average_val_loss = running_val_loss / len(val_loader)
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, zero_division=0)
#     recall = recall_score(all_labels, all_preds, zero_division=0)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)

#     return accuracy, average_val_loss, precision, recall, f1


# # train and validate model
# num_epochs = 20
# train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

# # Close the CSV file
# metrics_file.close()

# # save trained model to a file 
# torch.save(model.state_dict(), 'fire_cnn_16_.pth')
















#Load the test dataset
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Reload the model and load trained weights
model = FireCNN().to(device)
model.load_state_dict(torch.load(r'fire_cnn_16_11.pth'))
model.eval()  # Set the model to evaluation mode

# Evaluate the model on the test set
def evaluate_model_on_test_set(model, test_loader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # Move inputs to device
            outputs = model(inputs).squeeze()  # Forward pass
            preds = (outputs > 0.5).float()  # Threshold predictions
            all_preds.extend(preds.cpu().numpy())  # Collect predictions
            all_labels.extend(labels.numpy())  # Collect true labels

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    # Print metrics
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    
    # Return predictions, labels, and metrics
    return all_preds, all_labels, accuracy, precision, recall, f1

# Run the evaluation
all_preds, all_labels, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model_on_test_set(model, test_loader)

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["No Fire", "Fire"])

# Plot confusion matrix
plt.figure(figsize=(8, 6))  # Adjust figure size
disp.plot(cmap=plt.cm.Blues)  # Use a blue colormap

# Customize title and labels
plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted Labels', fontsize=20)
plt.ylabel('True Labels', fontsize=20)

# Show the plot
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, f1_score, confusion_matrix, classification_report
from load_data import create_dataset
from model import ECGNet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from focal_loss import FocalLoss

train_dataset, test_dataset, class_weights = create_dataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = ["Normal", "Ventricular", "Supraventricular", "Fusion", "Unknown"]
weights = torch.tensor(class_weights, dtype=torch.float)

model = ECGNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#criterion = nn.CrossEntropyLoss(weight=weights.to(device))

alpha = ([1.0, 1.5, 3.0, 3.5, 2.0])

criterion = FocalLoss(alpha=alpha, gamma=1.5, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(model, train_loader, criterion, optimizer, num_epochs=15):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predicted, average='macro', zero_division=0)
    
    plot_confusion_matrix(all_labels, all_predicted, classes)

    report = classification_report(all_labels, all_predicted, target_names=classes, digits=4)
    print(report)

    print(f'Test Loss: {total_loss/len(test_loader):.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, F1 Score: {f1:.4f}')

if __name__ == "__main__":
    num_epochs = 15
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    evaluate_model(model, test_loader, criterion)

    # Save the model
    torch.save(model.state_dict(), 'ecgnet_model.pth')
    print("Model saved as 'ecgnet_model.pth'")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = r'D:\Projects\tpd\data' 
BATCH_SIZE = 32
EPOCHS = 5 # As per project timeline for baseline training [cite: 79]
LEARNING_RATE = 0.001

def main():
    # 1. SETUP THE HARDWARE
    print("--- STEP 1: Checking Hardware ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. PREPARE THE DATA [cite: 28, 31, 36, 37]
    print("\n--- STEP 2: Preparing Data ---")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # [cite: 37]
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load images from your specific folders 
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), data_transforms['test'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
    }
    
    class_names = image_datasets['train'].classes
    print(f"✅ Data Loaded! Found {len(class_names)} classes.")

    # 3. BUILD THE MODEL (Transfer Learning) [cite: 39, 40, 42, 43]
    print("\n--- STEP 3: Building the Model (ResNet50) ---")
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze early layers to speed up Day 1 training [cite: 44]
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for your 10 tomato classes [cite: 12, 43]
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. TRAIN [cite: 45, 46, 47]
    print(f"\n--- STEP 4: Training for {EPOCHS} Epochs ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(image_datasets['train'])
        print(f'Training Loss: {epoch_loss:.4f}')

    # 5. FINAL EVALUATION (Macro F1 & Confusion Matrix) [cite: 50, 52, 53, 73]
    print("\n--- STEP 5: Evaluating Model ---")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calculate Metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n✅ FINAL EVALUATION REPORT:")
    print(f"Macro F1 Score: {macro_f1:.4f}") # Primary evaluation metric [cite: 73]
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    # Plot Confusion Matrix [cite: 53, 74]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Tomato Disease Detection')
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved as 'confusion_matrix2.png'")

    # 6. SAVE MODEL [cite: 68]
    torch.save(model.state_dict(), 'tomato_model2.pth')
    print("✅ Model weights saved as 'tomato_model2.pth'")

if __name__ == '__main__':
    main()
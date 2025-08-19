import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# Custom Dataset Class
class FERDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale for 48x48 images
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.image_paths = []
        self.labels = []
        for label, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    self.image_paths.append(os.path.join(emotion_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Load pre-trained ResNet18 and modify for grayscale (1 channel) and 7 classes
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 1
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 emotions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create datasets and data loaders
train_dataset = FERDataset(root_dir='data/train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = FERDataset(root_dir='data/test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop with validation
for epoch in range(10):
    # Training phase
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'emotion_model.pth')
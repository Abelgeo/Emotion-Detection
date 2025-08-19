import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class FERDataset(Dataset):
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

# Usage
train_dataset = FERDataset(root_dir='data/train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = FERDataset(root_dir='data/test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
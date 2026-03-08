import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.decom import Decom
from models.spam import LVRNet

class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.low_images = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg'))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg'))])
        self.transform = transform

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img = Image.open(os.path.join(self.low_dir, self.low_images[idx])).convert('RGB')
        high_img = Image.open(os.path.join(self.high_dir, self.high_images[idx])).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img

class IntegratedModel(nn.Module):
    def __init__(self, device):
        super(IntegratedModel, self).__init__()
        self.decom = Decom()
        self.spam = LVRNet(gps=3, blocks=16, bs=1)
        self.device = device

        # simple enhancement for R and L (can be replaced with more complex models)
        self.r_enhance = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

        self.l_enhance = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Retinex decomposition
        R, L = self.decom(x)

        # enhance components
        R_enhanced = self.r_enhance(R)
        L_enhanced = self.l_enhance(L)

        # combine
        combined = R_enhanced * L_enhanced

        # SPAM post-processing
        output = self.spam(combined)

        return output

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for low_img, high_img in pbar:
        low_img = low_img.to(device)
        high_img = high_img.to(device)

        optimizer.zero_grad()

        # forward pass
        output = model(low_img)

        # calculate loss
        loss = criterion(output, high_img)

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def main():
    # configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # paths (modify these to your actual paths)
    low_dir = "data/train/low"
    high_dir = "data/train/high"
    save_dir = "checkpoints"

    os.makedirs(save_dir, exist_ok=True)

    # dataset
    transform = transforms.Compose([
        transforms.Resize((256, 456)),
        transforms.ToTensor()
    ])

    print("loading dataset...")
    dataset = LowLightDataset(low_dir, high_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    print(f"dataset size: {len(dataset)}")

    # model
    print("creating model...")
    model = IntegratedModel(device).to(device)

    # loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # training loop
    num_epochs = 100
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # train
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Average loss: {avg_loss:.4f}")

        # save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)
            print(f"saved best model to {save_path}")

        # save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, save_path)
            print(f"saved checkpoint to {save_path}")

    print("training complete!")

if __name__ == "__main__":
    main()

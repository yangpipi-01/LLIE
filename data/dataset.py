import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir=None, transform=None, is_train=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.is_train = is_train

        # get low light images
        self.low_images = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg'))])

        # get high light images if available
        if high_dir and os.path.exists(high_dir):
            self.high_images = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg'))])
        else:
            self.high_images = None

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        # load low light image
        low_img = Image.open(os.path.join(self.low_dir, self.low_images[idx])).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)

        if self.is_train and self.high_images:
            # load high light image for training
            high_img = Image.open(os.path.join(self.high_dir, self.high_images[idx])).convert('RGB')
            if self.transform:
                high_img = self.transform(high_img)
            return low_img, high_img
        else:
            return low_img, self.low_images[idx]

def get_transform(img_size=(256, 456)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

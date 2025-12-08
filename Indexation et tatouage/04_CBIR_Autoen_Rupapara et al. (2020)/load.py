from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class CBIRDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # store absolute path
                    self.img_files.append(os.path.abspath(os.path.join(root, file)))
        if len(self.img_files) == 0:
            raise ValueError(f"No image files found in: {os.path.abspath(img_dir)}. Check the path or file extensions.")
        self.transform = transform if transform else self.do_transform()

    @staticmethod
    def do_transform():
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]   # already full path
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path
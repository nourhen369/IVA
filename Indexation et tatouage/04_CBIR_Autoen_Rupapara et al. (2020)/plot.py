import os
from typing import List, Optional

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import euclidean_distances

from load import CBIRDataset
from model import AutoEncoder


def plot(dataset_dir: str,
         model_path: str,
         query_image_path: str,
         top_k: int = 5,
         batch_size: int = 32,
         device: Optional[torch.device] = None,
         show: bool = True) -> List[str]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Load dataset and dataloader (CBIRDataset returns (tensor, path) per item)
    dataset = CBIRDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = AutoEncoder().to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract features
    features_list = []
    image_paths = []
    with torch.no_grad():
        for imgs, paths in dataloader:
            imgs = imgs.to(device)
            latent = model.encoder(imgs)  # expects encoder method in AutoEncoder
            features_list.append(latent.cpu().numpy())
            image_paths.extend(paths)

    if len(features_list) == 0:
        raise ValueError(f"No images found in dataset: {os.path.abspath(dataset_dir)}")

    features = np.vstack(features_list)

    # Ensure query is transformed like dataset
    query_img_pil = Image.open(query_image_path).convert("RGB")
    if hasattr(dataset, "transform") and dataset.transform is not None:
        query_tensor = dataset.transform(query_img_pil).unsqueeze(0).to(device)
    else:
        # fallback: convert to tensor and resize to 64x64
        from torchvision import transforms
        fallback_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        query_tensor = fallback_transform(query_img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        query_vec = model.encoder(query_tensor).cpu().numpy()

    dists = euclidean_distances(query_vec, features)
    indices = np.argsort(dists[0])[:top_k]
    results = [image_paths[i] for i in indices]

    # Plotting
    if show:
        plt.figure(figsize=(4, 4))
        plt.imshow(query_img_pil)
        plt.title("Query Image")
        plt.axis("off")

        plt.figure(figsize=(3 * top_k, 3))
        for i, img_path in enumerate(results):
            img = Image.open(img_path).convert("RGB")
            plt.subplot(1, top_k, i + 1)
            plt.imshow(img)
            plt.title(f"Similar {i+1}")
            plt.axis("off")
        plt.suptitle("Top Similar Images")
        plt.show()

    return results
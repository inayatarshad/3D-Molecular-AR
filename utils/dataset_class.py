# utils/dataset_class.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MoleculeDataset(Dataset):
    def __init__(self, labels_file, image_folders, vocab, max_length=220):
        self.labels_file = labels_file
        self.image_folders = image_folders
        self.vocab = vocab
        self.max_length = max_length
        self.data = self.load_data()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def load_data(self):
        df = pd.read_csv(self.labels_file)
        data = []

        for _, row in df.iterrows():
            filename = row['filename']
            smiles = row['smiles']
            image_path = self.find_image(filename)
            if image_path:
                data.append((image_path, smiles))
            else:
                print(f"⚠️ Warning: Missing image for {filename}, skipping.")

        print(f"✅ Loaded {len(data)} samples from {self.labels_file}")
        return data

    def find_image(self, filename):
        for folder in self.image_folders:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                return path
        return None

    def pad_smiles(self, smiles):
        encoded = self.vocab.encode(smiles)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length - 1] + [self.vocab.get_idx('<end>')]
        while len(encoded) < self.max_length:
            encoded.append(self.vocab.get_idx('<pad>'))
        return torch.tensor(encoded)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        image_path, smiles = self.data[idx]
        return self.load_image(image_path), self.pad_smiles(smiles)

    def __len__(self):
        return len(self.data)

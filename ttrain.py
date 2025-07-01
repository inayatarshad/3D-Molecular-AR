# train_model.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import os

from utils.dataset_class import MoleculeDataset
from utils.vocab import SMILESVocab


# ==== Model Definitions ====
class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        x = self.backbone(images).squeeze()
        return self.fc(x)


class GRUDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), 1)
        output, _ = self.gru(inputs)
        return self.fc(output[:, 1:])


def train(resume=False, model_path=None, start_epoch=0, total_epochs=140):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}\n")

    # Load vocab
    vocab = SMILESVocab()
    vocab.load_vocab("_vocab.pkl")
    vocab_size = len(vocab)

    # Dataset & DataLoader
    dataset = MoleculeDataset(
        "G:/PyCharm Projects/PROJECT/dataset/labels.csv",
        [
            "G:/PyCharm Projects/PROJECT/dataset/images_pubchem",
            "G:/PyCharm Projects/PROJECT/dataset/images_handdrawn"
        ],
        vocab=vocab
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Model init
    embed_size = 256
    hidden_size = 512
    encoder = CNNEncoder(embed_size).to(device)
    decoder = GRUDecoder(embed_size, hidden_size, vocab_size).to(device)

    # Load model weights if resuming
    if resume and model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        print(f"✅ Loaded saved model from {model_path}")

        # Try to get the last epoch from checkpoint if available
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
    else:
        print("🔄 Starting new training session")

    # Initialize optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token_to_idx['<pad>'])

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{total_epochs}]", leave=False)
        for imgs, smiles in loop:
            imgs, smiles = imgs.to(device), smiles.to(device)
            inputs, targets = smiles[:, :-1], smiles[:, 1:]

            features = encoder(imgs)
            outputs = decoder(features, inputs)

            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch + 1} Loss: {total_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': total_loss,
                'vocab': vocab.token_to_idx
            }, f"G:/PyCharm Projects/PROJECT/model/checkpoint_epoch_{epoch + 1}.pt")
            print(f"💾 Saved checkpoint at epoch {epoch + 1}")

    # Final save
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': total_epochs,
        'loss': total_loss,
        'vocab': vocab.token_to_idx
    }, "G:/PyCharm Projects/PROJECT/model/trained_model.pt")

    print("\n🎉 Training complete and model saved!")


if __name__ == "__main__":
    train(
        resume=True,
        model_path="G:/PyCharm Projects/PROJECT/model/trained_model.pt",
        start_epoch=110,  # Start from epoch 70
        total_epochs=140  # Train until epoch 140
    )
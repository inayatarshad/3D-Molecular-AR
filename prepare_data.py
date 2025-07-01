# prepare_data.py

import os
import pandas as pd
import pickle
from utils.data_loader import fetch_pubchem_images, fetch_handdrawn_images
from utils.vocab import SMILESVocab

def main():

    print("\n📥 Fetching PubChem images...")
    fetch_pubchem_images()

    print("\n✍️ Fetching Handdrawn images from HuggingFace...")
    fetch_handdrawn_images()

    print("\n🔠 Building vocabulary from SMILES...")
    df = pd.read_csv("G:/PyCharm Projects/PROJECT/dataset/labels.csv")
    smiles_list = df['smiles'].tolist()

    vocab = SMILESVocab()
    vocab.build_vocab(smiles_list)

    with open("_vocab.pkl", "wb") as f:
        pickle.dump(vocab.token_to_idx, f)

    print("\n✅ Data preparation complete.")

if __name__ == "__main__":
    main()

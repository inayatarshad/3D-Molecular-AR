# utils/data_loader.py

import os
import random
import csv
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import base64

def fetch_pubchem_images():
    os.makedirs("G:/PyCharm Projects/PROJECT/dataset/images_pubchem", exist_ok=True)
    cids = random.sample(range(1, 100_000_000), 1000)

    with open("G:/PyCharm Projects/PROJECT/dataset/labels.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'smiles'])

    for cid in cids:
        try:
            cmpd = pcp.Compound.from_cid(cid)
            smiles = cmpd.isomeric_smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(224, 224))
                img.save(f"G:/PyCharm Projects/PROJECT/dataset/images_pubchem/{cid}.png")

                with open("G:/PyCharm Projects/PROJECT/dataset/labels.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{cid}.png", smiles])
        except Exception as e:
            print(f"PubChem error CID {cid}: {e}")

def fetch_handdrawn_images():
    os.makedirs("G:/PyCharm Projects/PROJECT/dataset/images_handdrawn", exist_ok=True)
    dataset = load_dataset("Leonardo6/decimer", split="train")

    for idx, data in enumerate(dataset.select(range(1000))):
        try:
            # Extract the list of 'messages' from the dataset
            messages = data['messages']
            smiles = None

            for message in messages:
                if message['role'] == 'assistant':
                    smiles = message['content']
                    break

            if smiles:
                # Correct: Take the first image from the list
                image_data = data['images'][0]

                # image_data is already a PIL Image
                if isinstance(image_data, Image.Image):
                    img = image_data
                else:
                    raise ValueError(f"Unexpected image format at index {idx}: {type(image_data)}")

                # Save the image
                img.save(f"G:/PyCharm Projects/PROJECT/dataset/images_handdrawn/handdrawn_{idx}.png")

                # Save the corresponding SMILES
                with open("G:/PyCharm Projects/PROJECT/dataset/labels.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"handdrawn_{idx}.png", smiles])

        except Exception as e:
            print(f"Handdrawn error index {idx}: {e}")


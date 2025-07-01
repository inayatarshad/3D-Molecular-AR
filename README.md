# 🧪 3D Molecular AR — Augmented Reality for Molecular Geometry

An innovative AR application that superimposes 3D molecular structures over 2D chemical drawings to enhance visualization and understanding of molecular geometry.


---

## 🎯 Problem Statement

Traditional 2D diagrams fail to capture the spatial geometry of molecules — bond angles, orientations, and structures are hard to interpret. This leads to difficulties in understanding molecular chemistry, especially for students.

Our solution is an AR-based tool that uses a camera to detect 2D molecular structures and dynamically overlay 3D models on them, enabling real-time interaction, better comprehension, and an immersive learning experience.

---

## 📊 Data Preparation

We used two datasets:

1. **PubChem Sketcher Dataset**
   - 1000 hand-drawn molecular images
   - Each paired with its SMILES representation and name

2. **Decimer Dataset (via Hugging Face)**
   - 1000 molecular images
   - SMILES and molecule names included

Images and SMILES were cleaned, validated, and paired in `.csv` files. Vocabulary and token mappings were built using `SMILESVocab`.

---

## 🛠️ Technologies & Libraries Used

### Core
- `pandas`, `numpy`, `pickle`, `tqdm`, `requests`, `pillow`, `opencv-python`, `rembg`

### Deep Learning & Molecular Processing
- `torch`, `torchvision`, `rdkit`, `pubchempy`, `datasets`

### Custom Modules
- `fetch_pubchem_images` & `fetch_handdrawn_images` from `utils.data_loader`
- `SMILESVocab` from `utils.vocab`

---

## 🧪 Model Training

- **Encoder**: ResNet18 pretrained model extracts visual features from 2D images.
- **Decoder**: GRU network generates the SMILES string.
- **Training**: 
  - Teacher Forcing method
  - 80/20 train-test split
  - 140+ epochs, loss dropped from 150 to ~0.4

---

## 🚀 AR Implementation

- Users present a 2D molecule in front of a webcam.
- Pressing `S` scans the image and predicts its SMILES.
- SMILES is used to retrieve or generate a 3D molecular structure.
- The 3D model is rendered and overlaid via AR in real time.

---

## 🎨 Color Scheme (CPK Convention)

- **Atoms**:
  - Hydrogen: White
  - Carbon: Gray
  - Nitrogen: Blue
  - Oxygen: Red
  - Sulfur: Yellow
- **Bonds**:
  - Single: Gray, radius 0.07
  - Double: Yellow, radius 0.10
  - Triple: Red, radius 0.12

---

## 🔍 Demonstration Steps

1. Open the app and point your camera at a printed 2D molecule.
2. Press `S` to scan and analyze.
3. The 3D structure will appear over the image.
4. Use your mouse or gestures to rotate, zoom, and interact.

---

## 📈 Future Improvements

- Add interactivity (highlight atoms/bonds)
- Show molecule properties (charge, polarity)
- Allow user annotations
- Expand molecular database
- Optimize for low-end devices
- Improve speed & accuracy under various conditions

---




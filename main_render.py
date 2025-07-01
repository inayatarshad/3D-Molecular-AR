import os
os.environ['VTK_REMOTE_RENDER'] = '1'  # Forces software rendering if needed
os.environ['VTK_BACKEND'] = 'OpenGL2'  # Ensures correct backend
import cv2
import torch
import pickle
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torchvision.transforms as transforms
from PIL import Image
from ttrain import CNNEncoder, GRUDecoder
# Core VTK modules
# VTK imports - complete set
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkTubeFilter, vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import vtkmodules.vtkRenderingOpenGL2  # Required for OpenGL rendering
# ===== Configuration =====
VOCAB_PATH = "G:/PyCharm Projects/PROJECT/_vocab.pkl"
MODEL_PATH = "G:/PyCharm Projects/PROJECT/model/trained_model.pt"



# ===== VTK Molecule Renderer =====
class AccuratePubChemRenderer:
    def __init__(self, smiles, size=(400, 400)):
        self.smiles = smiles
        self.size = size
        self.mol = None
        self.renderer = vtkRenderer()
        self.last_pos = (0, 0)
        self.setup_renderer()
        self.is_rotating = False  # Track rotation state

    def setup_renderer(self):
        """Initialize VTK rendering pipeline"""
        import vtkmodules.vtkRenderingOpenGL2
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(*self.size)
        self.render_window.SetOffScreenRendering(1)

        # Setup interactor
        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)
        self.interactor.Initialize()

        self.window_to_image = vtkWindowToImageFilter()
        self.window_to_image.SetInput(self.render_window)
        self.window_to_image.SetScale(1)
        self.window_to_image.SetInputBufferTypeToRGBA()

    def handle_rotation(self, dx, dy):
        """Handle rotation based on mouse movement"""
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(-dx * 0.5)
        camera.Elevation(dy * 0.3)
        camera.OrthogonalizeViewUp()
        self.render_window.Render()

    def fetch_pubchem_3d(self):
        """Get EXACT 3D structure from PubChem with bond orders"""
        try:
            # Step 1: Get CID
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{self.smiles}/cids/JSON"
            response = requests.get(url, timeout=10)
            cid = response.json()['IdentifierList']['CID'][0]

            # Step 2: Get COMPLETE 3D record with bond orders
            sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d&response_type=save"
            response = requests.get(sdf_url, timeout=10)

            # Preserve all original data including bond orders
            self.mol = Chem.MolFromMolBlock(response.text, removeHs=False)

            # Verify we got bond orders
            if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in self.mol.GetBonds()):
                print("✅ Successfully loaded PubChem model with bond orders")
                return True
            else:
                print("⚠️ Warning: No multiple bonds detected in PubChem data")
                return False

        except Exception as e:
            print(f"❌ Failed to fetch PubChem 3D model: {e}")
            # Fallback to RDKit 3D generation
            print("🔄 Attempting local 3D generation...")
            self.mol = Chem.MolFromSmiles(self.smiles)
            if self.mol:
                self.mol = Chem.AddHs(self.mol)
                AllChem.EmbedMolecule(self.mol)
                AllChem.UFFOptimizeMolecule(self.mol)
                print("✅ Generated local 3D structure")
                return True
            return False

    def render_with_bond_orders(self):
        """Render with accurate bond types (single/double/triple)"""
        if not self.mol:
            return False

        conf = self.mol.GetConformer()

        # Create points (exact coordinates)
        points = vtkPoints()
        for i in range(self.mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            points.InsertNextPoint(pos.x, pos.y, pos.z)

        # Create separate bond sets by type
        single_bonds = vtkCellArray()
        double_bonds = vtkCellArray()
        triple_bonds = vtkCellArray()

        for bond in self.mol.GetBonds():
            # Create bond as two points
            bond_points = vtkIdList()
            bond_points.InsertNextId(bond.GetBeginAtomIdx())
            bond_points.InsertNextId(bond.GetEndAtomIdx())

            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.DOUBLE:
                double_bonds.InsertNextCell(bond_points)
            elif bond_type == Chem.BondType.TRIPLE:
                triple_bonds.InsertNextCell(bond_points)
            else:  # Single, aromatic, etc.
                single_bonds.InsertNextCell(bond_points)

        # Create molecular structures
        def create_bond_actor(bonds, radius, color):
            bond_poly = vtkPolyData()
            bond_poly.SetPoints(points)
            bond_poly.SetLines(bonds)

            tube = vtkTubeFilter()
            tube.SetInputData(bond_poly)
            tube.SetRadius(radius)
            tube.SetNumberOfSides(12)

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            return actor

        # Add bond actors with distinct styles
        self.renderer.AddActor(create_bond_actor(single_bonds, 0.07, (0.5, 0.5, 0.5)))  # Gray single
        self.renderer.AddActor(create_bond_actor(double_bonds, 0.1, (0.8, 0.8, 0)))  # Yellow double
        self.renderer.AddActor(create_bond_actor(triple_bonds, 0.12, (1, 0, 0)))  # Red triple

        # Create atoms with element-specific colors and sizes
        for atom in self.mol.GetAtoms():
            sphere = vtkSphereSource()
            sphere.SetRadius(self.get_atom_radius(atom.GetAtomicNum()))
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)

            # Create points for just this atom
            atom_points = vtkPoints()
            pos = conf.GetAtomPosition(atom.GetIdx())
            atom_points.InsertNextPoint(pos.x, pos.y, pos.z)

            poly = vtkPolyData()
            poly.SetPoints(atom_points)

            glyph = vtkGlyph3D()
            glyph.SetInputData(poly)
            glyph.SetSourceConnection(sphere.GetOutputPort())

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.get_atom_color(atom.GetAtomicNum()))
            self.renderer.AddActor(actor)

        self.renderer.SetBackground(0, 0, 0)
        self.renderer.ResetCamera()
        return True

    def get_atom_radius(self, atomic_num):
        """Get standard atom radii"""
        return {
            1: 0.2,  # H
            6: 0.4,  # C
            7: 0.4,  # N
            8: 0.4,  # O
            16: 0.5  # S
        }.get(atomic_num, 0.4)  # Default

    def get_atom_color(self, atomic_num):
        """Get CPK atom colors"""
        return {
            1: (1, 1, 1),  # White for H
            6: (0.4, 0.4, 0.4),  # Gray for C
            7: (0, 0, 1),  # Blue for N
            8: (1, 0, 0),  # Red for O
            16: (1, 1, 0)  # Yellow for S
        }.get(atomic_num, (0.8, 0.8, 0.8))  # Light gray default

    def get_rendered_image(self):
        """Get current view as numpy array"""
        self.render_window.Render()
        self.window_to_image.Modified()
        self.window_to_image.Update()

        vtk_image = self.window_to_image.GetOutput()
        arr = numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        arr = arr.reshape(self.size[1], self.size[0], -1)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


# ===== Molecule Detection Functions =====
def load_models():
    """Load trained models and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # Initialize models
    embed_size = 256
    hidden_size = 512
    encoder = CNNEncoder(embed_size).to(device)
    decoder = GRUDecoder(embed_size, hidden_size, vocab_size).to(device)

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab, inv_vocab, device


# Image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_smiles_from_frame(frame: np.ndarray, encoder, decoder, vocab, inv_vocab, device, max_len=100,
                              temperature=1.0) -> str:
    """Predict SMILES string from a camera frame"""
    try:
        image = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(image)
            input_token = torch.tensor([[vocab['<start>']]], device=device)
            smiles_indices = []
            hidden = None
            last_token = None
            repeat_count = 0

            for _ in range(max_len):
                embedded = decoder.embedding(input_token)
                if hidden is None:
                    output, hidden = decoder.gru(embedded)
                else:
                    output, hidden = decoder.gru(embedded, hidden)
                logits = decoder.fc(output.squeeze(1)) / temperature
                probs = torch.softmax(logits, dim=-1)
                predicted_idx = torch.multinomial(probs, num_samples=1)

                pred_id = predicted_idx.item()
                print(f"🔢 Predicted token: {inv_vocab.get(pred_id, '?')} ({pred_id})")

                if last_token == pred_id:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_token = pred_id
                if repeat_count > 10:
                    print("⚠️ Stuck in repeated token. Breaking early.")
                    break

                if pred_id == vocab['<end>']:
                    break

                smiles_indices.append(pred_id)
                input_token = predicted_idx.view(1, 1)

        smiles = ''.join(inv_vocab.get(i, '') for i in smiles_indices)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("❌ Predicted SMILES is invalid.")
            return None
        canonical_smiles = Chem.MolToSmiles(mol)
        print(f"✅ Canonical SMILES: {canonical_smiles}")
        return canonical_smiles

    except Exception as e:
        print(f"❌ Error predicting SMILES: {e}")
        return None


def draw_molecule_on_frame(frame: np.ndarray, mol_img: np.ndarray) -> np.ndarray:
    """Draw molecule on webcam frame"""
    h, w = mol_img.shape[:2]
    x, y = 50, 50

    # Make background transparent
    mask = cv2.cvtColor(mol_img, cv2.COLOR_BGRA2GRAY)
    _, alpha = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mol_img[:, :, 3] = alpha

    # Overlay on webcam
    alpha = mol_img[:, :, 3] / 255.0
    for c in range(3):
        frame[y:y + h, x:x + w, c] = (1 - alpha) * frame[y:y + h, x:x + w, c] + alpha * mol_img[:, :, c]

    return frame


# ===== Main Application =====
# ... (keep all your imports and configuration the same until the main() function)

def main():
    print("🚀 Starting AR Molecule Visualizer...")

    # Load models
    encoder, decoder, vocab, inv_vocab, device = load_models()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Webcam started. Press 's' to scan, 'q' to quit.")
    overlay_img = None
    current_smiles = None
    renderer = None

    # Mouse control callback
    def mouse_callback(event, x, y, flags, param):
        nonlocal overlay_img
        if renderer:
            camera = renderer.renderer.GetActiveCamera()

            if event == cv2.EVENT_LBUTTONDOWN:
                renderer.last_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                dx = x - renderer.last_pos[0]
                dy = y - renderer.last_pos[1]
                renderer.last_pos = (x, y)

                # Rotate based on mouse movement
                camera.Azimuth(-dx * 0.5)
                camera.Elevation(dy * 0.3)
                camera.OrthogonalizeViewUp()

                # Update the rendered image
                overlay_img = renderer.get_rendered_image()

    cv2.namedWindow("PubChem 3D - Drag to Rotate")
    cv2.setMouseCallback("PubChem 3D - Drag to Rotate", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Create a clean frame for display
        display_frame = frame.copy()

        if overlay_img is not None:
            # Make background transparent
            mask = cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2GRAY)
            _, alpha = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            overlay_img[:, :, 3] = alpha

            # Overlay on webcam
            h, w = overlay_img.shape[:2]
            x, y = 50, 50
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(3):
                display_frame[y:y + h, x:x + w, c] = (1 - alpha) * display_frame[y:y + h, x:x + w,
                                                                   c] + alpha * overlay_img[:, :, c]

        cv2.imshow("PubChem 3D - Drag to Rotate", display_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            print("🔍 Detecting molecule...")
            smiles = predict_smiles_from_frame(frame, encoder, decoder, vocab, inv_vocab, device)
            if smiles:
                print(f"✅ SMILES Detected: {smiles}")
                current_smiles = smiles
                renderer = AccuratePubChemRenderer(smiles)
                if not renderer.fetch_pubchem_3d():
                    print("⚠️ Using locally generated 3D structure")
                if renderer.render_with_bond_orders():
                    overlay_img = renderer.get_rendered_image()
                else:
                    print("❌ Failed to render molecule")
                    overlay_img = None
            else:
                print("❌ Molecule not recognized.")
                overlay_img = None

        elif key & 0xFF == ord('q'):
            print("👋 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
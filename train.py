from utils.process_ligand import processLigand
from utils.process_protein import processProtein
import torch_geometric as pyg
from torch_geometric.nn import PNAConv 
from models.model import AffinityLock
from utils.affinity_dict import affinitydict
from utils.preprocess_database import preprocess_database
import os 
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
import argparse

BATCH_SIZE = 32
# LIGAND/PROTEIN_3D: FALSE IF ONLY SCALAR, TRUE IF UTILISING 3D FEATURES
LIGAND_3D = False # 
PROTEIN_3D = False  #

class IndividualPTDataset(Dataset):
    def __init__(self, ligand_3d, protein_3d, folder_path="./database/processedfeatures"):
        super().__init__()
        self.folder = Path(folder_path)
        self.file_paths = sorted(list(self.folder.glob("*.pt")))
        self.ligand_3d = ligand_3d
        self.protein_3d = protein_3d
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.folder.resolve()}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        obj = torch.load(self.file_paths[idx], weights_only=False)
        # THEY ARE ALWAYS IN THE FORM OF (SCALAR FEATURES, VECTOR FEATURES)
        if not self.protein_3d:
            if isinstance(obj['protein'].x, (tuple, list)):
                obj['protein'].x = obj['protein'].x[0]
            obj['protein'].num_nodes = obj['protein'].x.size(0)
            if hasattr(obj['protein'], 'pos'):
                del obj['protein'].pos

        if not self.ligand_3d:
            if isinstance(obj['ligand'].x, (tuple, list)):
                obj['ligand'].x = obj['ligand'].x[0]
            obj['ligand'].num_nodes = obj['ligand'].x.size(0)
            if hasattr(obj['ligand'], 'pos'):
                del obj['ligand'].pos

        return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand_3d", action="store_true", default=LIGAND_3D)
    parser.add_argument("--protein_3d", action="store_true", default=PROTEIN_3D)
    args = parser.parse_args()
    print('ARGS:', args)
    print(f"Config -> LIGAND_3D: {args.ligand_3d} | PROTEIN_3D: {args.protein_3d}")
    #these take protein_pdb and ligand_pdb strings pointing to the relative .pdb file
    #preprocess this over the entire database and use torch.save 
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
    dict_ = affinitydict()
    dataset = IndividualPTDataset(
    ligand_3d=args.ligand_3d,
    protein_3d=args.protein_3d,
    folder_path="./database/processedfeatures"
    )
    print(f"Found {len(dataset)} preprocessed .pt files.")

    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=1,         
        persistent_workers=True
    )

    x = dataset[0]['ligand'].x
    ligand_dim = (x[0].shape[1], x[1].shape[1]) if args.ligand_3d else (x.shape[1], 0)

    x = dataset[0]['protein'].x
    protein_dim = (x[0].shape[1], x[1].shape[1]) if args.protein_3d else (x.shape[1], 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AffinityLock(ligand_dim, protein_dim, ligand_3d=args.ligand_3d, protein_3d=args.protein_3d).to(device)
    loss = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            ligand_b = batch['ligand'].to(device)
            protein_b = batch['protein'].to(device)
            y = batch['affinity'].to(device)

            optimiser.zero_grad()

            pred = model.forward(ligand_b, protein_b)
            loss_ = loss(pred, y.squeeze(-1))
            loss_.backward()
            optimiser.step()

            total_loss += loss_.item()

            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Step [{step+1}/{len(train_loader)}] | Loss: {loss_.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{num_epochs}] Complete | Avg Loss: {avg_loss:.4f} ---")

    

if __name__ == "__main__":
    main()

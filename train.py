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

class IndividualPTDataset(Dataset):
    def __init__(self, folder_path="./database/processedfeatures"):
        super().__init__()
        self.folder = Path(folder_path)
        self.file_paths = sorted(list(self.folder.glob("*.pt")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.folder.resolve()}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Loads single dictionary object on demand
        obj = torch.load(self.file_paths[idx], weights_only=False)
        return obj

def main():
    #these take protein_pdb and ligand_pdb strings pointing to the relative .pdb file
    #preprocess this over the entire database and use torch.save 
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
    dict_ = affinitydict()
    dataset = IndividualPTDataset("./database/processedfeatures")
    print(f"Found {len(dataset)} preprocessed .pt files.")

    train_loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,         
        persistent_workers=True
    )

    ligand_dim = dataset[0]['ligand'].x.shape[1]
    protein_xscalar, protein_xvec = dataset[0]['protein'].x
    protein_dim = protein_xscalar.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AffinityLock(ligand_dim, protein_dim).to(device)
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
            print('predicted output =', pred)
            loss_ = loss(pred, y)

            loss_.backward()
            optimiser.step()

            total_loss += loss_.item()

            if (step + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Step [{step+1}/{len(train_loader)}] | Loss: {loss_.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{num_epochs}] Complete | Avg Loss: {avg_loss:.4f} ---")

    

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GPSConv

class AffinityLock(nn.Module):
    def __init__(self, ligand_dim, protein_dim, hidden_dim=64):
        #https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
        super().__init__()
        
        #ligand encoder
        self.ligconv1 = GPSConv(ligand_dim, hidden_dim) #gpsconv
        self.ligconv2 = GPSConv(hidden_dim, hidden_dim) #gpsconv

        #protein encoder 
        self.protconv1 = GPSConv(protein_dim, hidden_dim)  #gpsconv
        self.protconv2 = GPSConv(hidden_dim, hidden_dim)   #gpsconv

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lig_graph, prot_graph):
        pass
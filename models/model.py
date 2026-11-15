import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GPSConv, GCNConv, global_mean_pool

def encodeLigand(in_dim, is3D, hidden_dim=64, num_layers=2):
    return Ligand2DEncoder(in_dim) if not is3D else 1

def encodeProtein(in_dim, is3D, hidden_dim=64, num_layers=2):
    return Protein2DEncoder(in_dim) if not is3D else 1

class Ligand2DEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        in_dim = in_dim[0] # SINCE IN DIM = (SCALAR, 0), JUST PULL OUT SCALAR
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim))
            for _ in range(num_layers)])
        
    def forward(self, graph):
        # since ligand is 2D, only scalar features; as its (scalar, 0) 
        x = graph.x
        x = self.proj(x)
        for conv in self.convs:
            x = conv(x, graph.edge_index, graph.batch)
        return global_mean_pool(x, graph.batch)

class Protein2DEncoder(nn.Module):
    # GCN conv is graph convolutional network that conducts message passing to features,
    # its wrapped in GPS Conv which runs gcn conv in parallel with a global attention mechanism
    def __init__(self, in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        in_dim = in_dim[0]
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim))
            for _ in range(num_layers)])
        
    def forward(self, graph):
        # since protein is 2D, only scalar features, as its (scalar, 0)
        x = graph.x
        x = self.proj(x)
        for conv in self.convs:
            x = conv(x, graph.edge_index, graph.batch)
        return global_mean_pool(x, graph.batch)

class AffinityLock(nn.Module):
    def __init__(self, ligand_dim, protein_dim, ligand_3d=False, protein_3d=False, hidden_dim=64):
        #https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
        super().__init__()

        # ligand dimension is typically much smaller than protien, idea here is the first layer is that this 
        # layer takes as input features and cast them to a length 64 (hidden dim) tensor
        self.lig_encoder = encodeLigand(ligand_dim, ligand_3d)
        self.prot_encoder = encodeProtein(protein_dim, protein_3d)

        print(self.lig_encoder, 'prot=', self.prot_encoder)

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lig_graph, prot_graph):
        ligand_pool = self.lig_encoder(lig_graph)
        protein_pool = self.prot_encoder(prot_graph)
        combined = torch.cat([ligand_pool, protein_pool], dim=1)
        pred = self.layers(combined)
        return pred.squeeze(-1)

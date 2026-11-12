import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GPSConv, GCNConv, global_mean_pool

class AffinityLock(nn.Module):
    def __init__(self, ligand_dim, protein_dim, hidden_dim=64):
        #https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
        super().__init__()

        self.lig_proj = nn.Linear(ligand_dim, hidden_dim)
        self.prot_proj = nn.Linear(protein_dim, hidden_dim)
        
        #ligand encoder
        self.ligconv1 = GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim)) 
        self.ligconv2 = GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim)) 

        #protein encoder 
        self.protconv1 = GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim))  
        self.protconv2 = GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim))

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lig_graph, prot_graph):
        ligand_nodes, ligand_edges, batch_lig = lig_graph.x, lig_graph.edge_index, lig_graph.batch
        protein_nodes, protein_edges, batch_prot = prot_graph.x, prot_graph.edge_index, prot_graph.batch

        ligand_nodes = self.lig_proj(ligand_nodes)
        ligand_nodes = self.ligconv1(ligand_nodes, ligand_edges, batch_lig)
        ligand_nodes = self.ligconv2(ligand_nodes, ligand_edges, batch_lig)
        ligand_pool = global_mean_pool(ligand_nodes, batch_lig)

        #USE ONLY SCALAR FEATURES FROM PROTEIN (AS WE ALSO HAVE VEC FEATURES BUT NOT IMPLEMENTED 3D YET)
        protein_nodes = protein_nodes[0]
        protein_nodes = self.prot_proj(protein_nodes)
        protein_nodes = self.protconv1(protein_nodes, protein_edges, batch_prot)
        protein_nodes = self.protconv2(protein_nodes, protein_edges, batch_prot)
        protein_pool = global_mean_pool(protein_nodes, batch_prot)


        combined = torch.cat([ligand_pool, protein_pool], dim=1)
        pred = self.layers(combined)
        return pred.squeeze(-1)
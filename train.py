from utils.process_ligand import processLigand
from utils.process_protein import processProtein
import torch_geometric as pyg
from torch_geometric.nn import PNAConv 
from models.model import AffinityLock
from utils.affinity_dict import affinitydict
from utils.preprocess_database import preprocess_database
import os 


def main():
    #these take protein_pdb and ligand_pdb strings pointing to the relative .pdb file
    #preprocess this over the entire database and use torch.save 
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
    dict_ = affinitydict()
    # processed_file = "./database/processedfeatures.pt"

    # THIS NEEDS TO BE RERUN EVERYTIME FEATURES ARE CHANGED!
    if os.path.exists(processed_file):
        print("Loading preprocessed dataset from disk...")
        dataset = torch.load(processed_file)
    else:
        print("Preprocessed dataset not found. Building it now (this may take a while)...")
        preprocess_database(dict_, "./database/P-L", "./database/processedfeatures.pt")

    l = processLigand() # use .graph attribute to use PyG graph
    p = processProtein() # use .graph as above

    #.x, .edge_attr, .edge_index
    ligand = l.graph
    protein = p.graph 

    print(ligand, protein)
    ligand_dim = ligand.x.shape[1]

    #only utilising scalar features from protein atm, ligand is only scalar features anyways


    protein_xscalar, protein_xvec = protein.x

    protein_dim = protein_xscalar.shape[1]

    model = AffinityLock(ligand_dim, protein_dim)
    loss = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # target_affinity = affinitydict['4b60']
    # print(target_affinity)

    

if __name__ == "__main__":
    main()
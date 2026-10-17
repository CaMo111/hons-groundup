from utils.process_ligand import processLigand
from utils.process_protein import processProtein
import torch_geometric as pyg

def main():
    ligand = processLigand() # use .graph attribute to use PyG graph
    protein = processProtein() # use .graph as above

    print(ligand, protein)
    #lig_enc = pyg.
    

if __name__ == "__main__":
    main()

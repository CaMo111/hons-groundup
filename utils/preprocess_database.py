from collections import defaultdict
import mdtraj as md
import numpy as np
import ipdb
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
import torch_geometric as pyg
import os 
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.process_ligand import processLigand
from utils.process_protein import processProtein
from utils.affinity_dict import affinitydict

def preprocess_database(dict_, database_dir, save_path, save=False):
    # create a global .pt dataset that runs features for all 
    # only preprocess all features if not globally done alr
    # then can use dataloader 
    # if not os.path.exists(save_path):
    #     print('save path does not exist' )
    # else:
    #     print('save path does exist: ', save_path)

    # if not os.path.exists(save_path):
    # Utilise save keyword to recalculate features
    if save:
        aff_dict = dict_
        base_path = Path(database_dir)

        for year_dir in base_path.iterdir():
            for pdb_dir in year_dir.iterdir():

                pdb_id = pdb_dir.name
                protein_file = pdb_dir / f"{pdb_id}_protein.pdb"
                ligand_file = pdb_dir / f"{pdb_id}_ligand.mol2"

                print(pdb_id, protein_file, ligand_file)
                
                try:
                    l = processLigand(ligand_file)
                    p = processProtein(protein_file)
                    target = torch.tensor([aff_dict[pdb_id]], dtype=torch.float)
                    obj = {
                        'ligand': l.graph,
                        'protein': p.graph,
                        'affinity': target,
                        'pdb_id': pdb_id
                    }         
                    _save = save_path + f"{pdb_id}.pt"
                    torch.save(obj, _save)         

                except Exception as e:
                    print(f"failed to process {pdb_id}: {e}")

        print(f"successfully processed protein-ligand pairs")

if __name__ == "__main__":
    # RUN USING THIS COMMAND: python -m utils.preprocess_database
    preprocess_database(affinitydict(), './database/sample/', './database/sample/test/', save=True)
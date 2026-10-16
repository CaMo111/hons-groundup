from collections import defaultdict
import mdtraj as md
import numpy as np
import ipdb
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
from ligand_features import compute_ligand_edge_features, compute_ligand_node_features 

class processLigand():

    def __init__(self, lignad_pdb='../P-L/2011-2019/4b60/4b60_ligand.mol2', isSmileStringInput=False):
        self.pathstr_ = lignad_pdb
        print('init molecule')
        if isSmileStringInput:
            raise ValueError(f"CURRENTLY CASTING MOL2 -> RDKIT MOL, FOR ACTUAL IMPLEMENTATION MUST BE ABLE TO PARSE SMILE STR")
        self.mol = Chem.MolFromMol2File(self.pathstr_, removeHs=False)
        if self.mol is None:
            raise ValueError(f"Could not parse MOL2 file: {self.pathstr_}")

        #each ligand has X amount of atoms. We cast this to a 2d Graph of nodes and edges. Nodes represent individual atom features
        # 78 per atom as per graphDTA. we then construct edges which represent the chemical bond between the atoms.
        self.mol_nodes = compute_ligand_node_features(self.mol)
        self.mol_edges = compute_ligand_edge_features(self.mol)
        

if __name__ == "__main__":
    pdb = processLigand()
    print('ligand nodes', pdb.mol_nodes)
    print('ligand edges', pdb.mol_edges)
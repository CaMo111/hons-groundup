from collections import defaultdict
import mdtraj as md
import numpy as np
import ipdb
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
from utils.ligand_features import compute_ligand_edge_features, compute_ligand_node_features 
from utils.construct_pyg import construct_graph

class processLigand():

    def __init__(self, lignad_pdb='./P-L/2011-2019/4b60/4b60_ligand.mol2', isSmileStringInput=False):
        self.pathstr_ = lignad_pdb
        print('init molecule')
        if isSmileStringInput:
            raise ValueError(f"CURRENTLY CASTING MOL2 -> RDKIT MOL, FOR ACTUAL IMPLEMENTATION MUST BE ABLE TO PARSE SMILE STR")
        self.mol = Chem.MolFromMol2File(self.pathstr_, removeHs=False)
        if self.mol is None:
            raise ValueError(f"Could not parse MOL2 file: {self.pathstr_}")

        #each ligand has X amount of atoms. We cast this to a 2d Graph of nodes and edges. Nodes represent individual atom features
        # 78 per atom as per graphDTA. we then construct edges which represent the chemical bond between the atoms.
        
        #ligand nodes shape= (number_atoms, 78) -> 78 scalar features per ligand atom
        self.mol_nodes = compute_ligand_node_features(self.mol)
        
        #ligand edges shape= (198, 198, 9) -> 9 scalar features (bond features) from one atom to another, if no bond then NaN (majority are nan), features length 9 
        # ie we know bond between atom idx 0 and atom idx 23 therefore mol_edges[1,24] = [1.0,0,0,0,0,0,0,1.0,1.0]
        self.mol_edges = compute_ligand_edge_features(self.mol)
        self.graph = construct_graph(self.mol_nodes, self.mol_edges)
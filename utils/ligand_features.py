import numpy as np
import torch
import scipy.spatial.distance as sp_dist
import ipdb
from collections import defaultdict

def map_intdict_to_onehot(in_dict, add_other=False):
    """
    Map values to one-hot vectors
    """
    num_elements = len(in_dict)
    if(add_other):
        num_elements += 1
    
    onehot_dict = {}
    for k, v in in_dict.items():
        onehot = [0] * num_elements
        onehot[v] = 1
        onehot_dict[k] = onehot

    if(add_other):
        onehot_other = ([0] * (num_elements-1)) + [1]
        onehot_dict = defaultdict(lambda: onehot_other, onehot_dict)

    return onehot_dict

# i assume we should use list because we want constant time 
STEREO_CONFIGS = [
    'STEREONONE',
    'STEREOANY',
    'STEREOZ',
    'STEREOE',
    'STEREOCIS',
    'STEREOTRANS',
]
SMILES_STEREO_MAP = {j: i for i, j in enumerate(STEREO_CONFIGS)}
SMILES_STEREO_MAP = map_intdict_to_onehot(SMILES_STEREO_MAP, add_other=True)

#https://github.com/595693085/DGraphDTA/tree/master
# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

#https://github.com/595693085/DGraphDTA/tree/master
def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# https://github.com/595693085/DGraphDTA/tree/master
def atom_features(atom):
    # 44 +11 +11 +11 +1 
    # 44 atom symbols
    # 11 dims for one-hot vector of the number of directly bonded neighbors ($0$ to $10$).
    # 11 dims for total H
    # 11 dims for implicit valence
    # 1 dims for Aromaticity
    # total = 78
    encoding = (one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
    # cast true -> 1 false -> 0
    return np.array(encoding, dtype=np.float32)

# https://github.com/595693085/DGraphDTA/tree/master
def compute_ligand_node_features(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)

    return features

#https://github.com/rachitk/caster-dta/blob/main/utils/create_smiles_features.py    
def compute_ligand_edge_features(mol):    
    no_atoms = mol.GetNumAtoms()
    #print(no_atoms)
    # im a bit sus of this right now; creating 70x70 2d array --> I assume nan will filter this out 
    edge_features = np.empty((no_atoms, no_atoms, 9)) # 6 one hot + 2 for conjugated and ring
    edge_features.fill(np.nan)

    for bond in mol.GetBonds():
        bond_feats = []
        # One hot encoded features: HAVE YET TO IMPLEMENT THIS 
        #bond_feats.extend(smiles_maps.SMILES_STEREO_MAP[str(bond.GetStereo())])
        bond_feats.extend(SMILES_STEREO_MAP[str(bond.GetStereo())])
        bond_feats.append(int(bond.GetIsConjugated()))
        bond_feats.append(int(bond.IsInRing()))
        bond_feats = np.asarray(bond_feats, dtype=np.float32)

        edge_features[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_feats
        edge_features[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_feats

        # USE TO TEST ABOVE FEATS
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    #print(edges)
    #print(edge_features[1][24:34])
    #print("Atom 1 -> Atom 24 features:", edge_features[1, 24])
    #print("Atom 24 -> Atom 1 features:", edge_features[24, 1])

    return edge_features
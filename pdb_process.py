import mdtraj as md
import numpy as np
import ipdb
from protein_features import compute_residue_node_features

PROTEIN_3LETTER_1LETTER_MAP = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
    'UNK': 'X',
}

PROTEIN_1LETTER_3LETTER_MAP = {v: k for k, v in PROTEIN_3LETTER_1LETTER_MAP.items()}
PROTEIN_1LETTER_INT_MAP = {aa: i for i, aa in enumerate(PROTEIN_3LETTER_1LETTER_MAP.values())}
PROTEIN_INT_1LETTER_MAP = {v: k for k, v in PROTEIN_1LETTER_INT_MAP.items()}

def process_pdb(pdb_path):
    md_pdb = md.load(pdb_path)
    return md_pdb

def main():
    # load protein file and look at atom coordinates
    protein_frame = process_pdb('2011-2019/3rvi/3rvi_protein.pdb')
    print(protein_frame)

    # retain only backbone atoms (ie backbones atoms)
    idx = protein_frame.top.select("is_protein and (name N or name CA or name C or name O)")
    protein_backbones = protein_frame.atom_slice(idx)
    print(protein_backbones)

    # if protein has more than one frame (this often results from multiple conformers being deposited, but generally conformers
    # are very similar to one another - so a consistent approach works best here)
    if protein_backbones.n_frames > 1:
        protein_backbones = protein_backbones[0]

    # if first and last residues dont have all 4 atoms in (N, CA, C, O) then clip them off
    n_atoms = protein_backbones.n_atoms
    atoms = [atom.name for atom in protein_backbones.top.atoms]
    atom_concat = ''.join(atoms)
    expected='NCACO'
    first_ind = atom_concat.find(expected)
    last_ind = atom_concat[::-1].find(expected[::-1])
    protein_backbones = protein_backbones.atom_slice(range(first_ind, n_atoms-last_ind))

    # get number of residues in protein and corresponding coordinates
    # (n_residues, n_atoms_per_residue, 3))
    # THE IDEA IS THAT WE HAVE A STRUCTURE WHERE WE CAN ACCESS RESIDUE -> ATOM -> COORDINATES
    # IE RES_COORDS[0] = FIRST RESIDUE, RES_COORDS[0][0] = THE X,Y,Z COORDINATE OF THE FIRST ATOM OF THE RESIDUE
    atom_coords = protein_backbones.xyz
    n_residues = protein_backbones.top.n_residues
    res_coords = atom_coords.reshape(n_residues,4,3)

    #cast to angstrom, since we calculate within 3-4 angstrom for features
    res_coords = res_coords * 10

    #convert residue names to integers for ML 

    # Get the actual residue identities for each atom as an index
    # (these will be either node types or a particular feature)
    res_names = [res.name for res in protein_backbones.top.residues]
    res_1letter_names = [PROTEIN_3LETTER_1LETTER_MAP[res] for res in res_names]
    res_idents = [PROTEIN_1LETTER_INT_MAP[res] for res in res_1letter_names]
    print(res_idents)

    print(len(res_idents))
    # pass in the protein (residue/atoms/coords) with the sequence identities to construct 3D GNN Node features
    compute_residue_node_features(res_coords, res_idents)

if __name__ == "__main__":
     main()

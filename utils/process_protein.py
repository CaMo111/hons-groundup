from collections import defaultdict
import mdtraj as md
import numpy as np
import ipdb
from protein_features import compute_residue_node_features, compute_residue_edge_features

def normalize_aa_dict(d, X_val=0.0):
    """
    Normalize a dictionary to be between 0 and 1
    """
    min_val = min(d.values())
    max_val = max(d.values())
    diff = max_val - min_val

    out_d = {k: (v - min_val) / diff for k, v in d.items()}

    if(X_val is not None):
        if(X_val == 'mean'):
            X_mean = sum(out_d.values()) / len(out_d)
            out_d['X'] = X_mean
        else:
            out_d['X'] = X_val
        
    return out_d

class processProtein():

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
    PROTEIN_3LETTER_1LETTER_MAP = defaultdict(lambda: 'X', PROTEIN_3LETTER_1LETTER_MAP)

    AA_WEIGHTS = {
    'A': 89.09,
    'C': 121.16,
    'D': 133.10,
    'E': 147.13,
    'F': 165.19,
    'G': 75.07,
    'H': 155.15,
    'I': 131.17,
    'K': 146.19,
    'L': 131.17,
    'M': 149.21,
    'N': 132.12,
    'P': 115.13,
    'Q': 146.14,
    'R': 174.20,
    'S': 105.09,
    'T': 119.12,
    'V': 117.15,
    'W': 204.23,
    'Y': 181.19,
}
    AA_WEIGHTS = normalize_aa_dict(AA_WEIGHTS)

    AA_PKAS = {
    'A': 2.33,
    'C': 1.91,
    'D': 1.95,
    'E': 2.16,
    'F': 2.18,
    'G': 2.34,
    'H': 1.70,
    'I': 2.26,
    'K': 2.15,
    'L': 2.32,
    'M': 2.16,
    'N': 2.16,
    'P': 1.95,
    'Q': 2.18,
    'R': 2.03,
    'S': 2.13,
    'T': 2.20,
    'V': 2.27,
    'W': 2.38,
    'Y': 2.24,
    }
    AA_PKAS = normalize_aa_dict(AA_PKAS)

    # Residue pKb values
    # from the CRC Handbook pKb column
    # Represents the -NH2 acid dissociation constant
    AA_PKBS = {
    'A': 9.71,
    'C': 10.28,
    'D': 9.66,
    'E': 9.58,
    'F': 9.09,
    'G': 9.58,
    'H': 9.09,
    'I': 9.60,
    'K': 9.16,
    'L': 9.58,
    'M': 9.08,
    'N': 8.73,
    'P': 10.47,
    'Q': 9.00,
    'R': 9.00,
    'S': 9.05,
    'T': 8.96,
    'V': 9.52,
    'W': 9.34,
    'Y': 9.04,
    }
    AA_PKBS = normalize_aa_dict(AA_PKBS)


    # Residue pKc values
    # from the CRC Handbook pKc column
    # Represents the acid dissociation constant of 
    # other molecule groups if present (likely functional group)
    AA_PKCS = {
    'A': 0.0,
    'C': 8.14,
    'D': 3.71,
    'E': 4.15,
    'F': 0.0,
    'G': 0.0,
    'H': 6.04,
    'I': 0.0,
    'K': 10.67,
    'L': 0.0,
    'M': 0.0,
    'N': 0.0,
    'P': 0.0,
    'Q': 0.0,
    'R': 12.10,
    'S': 0.0,
    'T': 0.0,
    'V': 0.0,
    'W': 0.0,
    'Y': 10.10,
    }
    AA_PKCS = normalize_aa_dict(AA_PKCS)


    # Residue pI values
    # from the CRC Handbook pKi column
    # Represents the pH at the isoelectric point
    AA_PKIS = {
    'A': 6.00,
    'C': 5.07,
    'D': 2.77,
    'E': 3.22,
    'F': 5.48,
    'G': 5.97,
    'H': 7.59,
    'I': 6.02,
    'K': 9.74,
    'L': 5.98,
    'M': 5.74,
    'N': 5.41,
    'P': 6.30,
    'Q': 5.65,
    'R': 10.76,
    'S': 5.68,
    'T': 5.60,
    'V': 5.96,
    'W': 5.89,
    'Y': 5.66,
    }
    AA_PKIS = normalize_aa_dict(AA_PKIS)

    # Residue hydrophobicity
    # Using the Wimley-White scale based on results from:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9475378/
    # and https://biolres.biomedcentral.com/articles/10.1186/s40659-016-0092-5
    # showing that either W-W is the best or all of the scales are similar
    # W-W itself is provided here: https://www.nature.com/articles/nsb1096-842
    # We use the values as defined here: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/hydrophob.html
    # which states: "Values for the ionized forms of asp, glu, his are used here" and "More positive means more hydrophobic"
    AA_HYDROPHOB = {
    'A': -0.17,
    'C': 0.24,
    'D': -1.23,
    'E': -2.02,
    'F': 1.13,
    'G': -0.01,
    'H': -0.96,
    'I': 0.31,
    'K': -0.99,
    'L': 0.56,
    'M': 0.23,
    'N': -0.42,
    'P': -0.45,
    'Q': -0.58,
    'R': -0.81,
    'S': -0.13,
    'T': -0.14,
    'V': -0.07,
    'W': 1.85,
    'Y': 0.94,
    }
    AA_HYDROPHOB = normalize_aa_dict(AA_HYDROPHOB)

    # Residue alphaticity
    # Using the CRC Handbook 
    # and also https://pubs.acs.org/doi/10.1021/bi0105330
    # will also consider proline as aliphatic based on https://pubmed.ncbi.nlm.nih.gov/37960878/
    # and https://pubmed.ncbi.nlm.nih.gov/37056722/
    # and glycine? https://pubmed.ncbi.nlm.nih.gov/34270256/
    ALI_LIST = ['A', 'G', 'I', 'L', 'P', 'V']
    AA_ALIPHATIC = defaultdict(lambda: 0, {aa: 1 for aa in ALI_LIST})

    # Residue aromaticity
    # Using the CRC Handbook
    # and also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3530875
    ARO_LIST = ['F', 'H', 'W', 'Y']
    AA_AROMATIC = defaultdict(lambda: 0, {aa: 1 for aa in ARO_LIST})


    # Acidic residues
    # Using the CRC Handbook
    # and also https://pubmed.ncbi.nlm.nih.gov/38073323/
    ACID_LIST = ['D', 'E']
    AA_ACIDIC = defaultdict(lambda: 0, {aa: 1 for aa in ACID_LIST})


    # Basic residues
    # Using the CRC Handbook
    # and also https://pubmed.ncbi.nlm.nih.gov/21623415/
    BASIC_LIST = ['H', 'K', 'R']
    AA_BASIC = defaultdict(lambda: 0, {aa: 1 for aa in BASIC_LIST})


    # Polar neutral residues
    # Using the CRC Handbook
    # and also https://search.worldcat.org/title/297392560 (a textbook)
    POLAR_NEUTRAL_LIST = ['N', 'Q', 'S', 'T']
    AA_POLAR_NEUTRAL = defaultdict(lambda: 0, {aa: 1 for aa in POLAR_NEUTRAL_LIST})


    def __init__(self, protein_pdb='../P-L/2011-2019/4b60/4b60_protein.pdb'):
        self.pathstr_ = protein_pdb
        self.protein_frame = md.load(self.pathstr_) # frame which has atoms etc.
        
        # after preprocessing init
        self.res_coords = None # [n_res, 4, 3] -> [residue1, ['N', 'CA', 'C', 'O'][X,Y,Z]]
        self.res_idents = None # protein string
        self.nodes = None # 3D GNN features for nodes
        self.edges = None # 3D GNN features for edges 
        self.no_res = None

if __name__ == "__main__":
    pdb = processProtein()

    protein_frame = pdb.protein_frame
    # retain only backbone atoms (ie backbones atoms)
    idx = protein_frame.top.select("is_protein and (name N or name CA or name C or name O)")
    protein_backbones = protein_frame.atom_slice(idx)

    # if protein has more than one frame (this often results from multiple conformers being deposited, but generally conformers
    # are very similar to one another - so a consistent approach works best here)
    if protein_backbones.n_frames > 1:
        protein_backbones = protein_backbones[0]

    pdb.protein_frame = protein_backbones

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
    res_names = [res.name for res in protein_backbones.top.residues]
    res_1letter_names = [pdb.PROTEIN_3LETTER_1LETTER_MAP[res] for res in res_names]
    res_idents = [pdb.PROTEIN_1LETTER_INT_MAP[res] for res in res_1letter_names]


    pdb.res_coords = res_coords
    pdb.res_idents = res_idents
    # pass in the protein (residue/atoms/coords) with the sequence identities to construct 3D GNN Node features, as well as the specific protein we are dealing with
    # compute and collect node features 
    pdb.nodes = compute_residue_node_features(pdb.res_coords, pdb.res_idents, pdb) 
    
    #print('residuefeaturenodes', pdb.nodes)
    # edge features split into SCALAR FEATURES then VECTOR FEATURES (both 387x387)
    pdb.edges = compute_residue_edge_features(pdb.res_coords, pdb.res_idents, pdb)

    print('protein nodes=', pdb.nodes) 
    print('protein edges=', pdb.edges)

    #total_nans = np.isnan(pdb.edges[0]).sum() + np.isnan(pdb.edges[1]).sum()
    #total_values = pdb.edges[0].size + pdb.edges[1].size
    #fraction_all_nans = total_nans / total_values

    #print(f"Edge Features: {total_nans}")
    #print(f"Total NaNs: {total_nans}")
    #print(f"Total Values: {total_values}")
    #print(f"Overall Fraction: {fraction_all_nans:.6f} ({fraction_all_nans * 100:.2f}%)")

    #print(pdb.edges[0][0][1])
    #print(pdb.edges[1][0][1][0])

    #print('protein edge features:', pdb.edges)

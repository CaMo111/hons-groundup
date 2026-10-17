# TAKES AS INPUT Ligand PyG graph
#LIGAND NODES (ATOMS) -> (number_atoms, 78) -> 78 scalar features per ligand atom
# AND LIGAND BONDS -> (number_atoms, number_atoms, 9) -> 9 scalar features (bond features) from one atom to another, if no bond then NaN (majority are nan), features length 9 
    
def 
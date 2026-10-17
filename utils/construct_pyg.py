# cast node + edge scalar/vec features into a PYG Graph. This is necessary as all the NaNs will be chopped off
import torch
import torch_geometric as pyg

def construct_graph(nodes, edges):
    # ligand features are both numpy arrays, but protein features (nodes/edges) are both tuples 
    #for protein nodes: FUNCTION TAKEN FROM CASTER-DTA
    if(isinstance(nodes, tuple)):
        node_features = tuple(torch.tensor(feat, dtype=torch.float32) for feat in nodes)
        n_residues = node_features[0].shape[0]
    #for ligand nodes
    else:
        node_features = torch.tensor(nodes, dtype=torch.float32)
        n_residues = node_features.shape[0]

    edge_index = torch.tensor([[i, j] for i in range(n_residues) for j in range(n_residues)], dtype=torch.long).t().contiguous()

    # protein edges
    if(isinstance(edges, tuple)):
        edge_attr = tuple(torch.tensor(feat, dtype=torch.float32).flatten(0,1) for feat in edges)
        edge_nan_checker = edge_attr[0]
    #ligand edges
    else:
        edge_attr = torch.tensor(edges, dtype=torch.float32).flatten(0,1)
        edge_nan_checker = edge_attr

    # CUT OUT NAN (THIS IS SUPER IMPORTANT):
    keep_indices = (~torch.isnan(edge_nan_checker).all(axis=-1))
    edge_index = edge_index[:,keep_indices]
    if(isinstance(edges, tuple)):
        edge_attr = tuple(feat[keep_indices,:] for feat in edge_attr)
    else:
        edge_attr = edge_attr[keep_indices,:]
    
    pyg_data = pyg.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data
    """
    FOR PROTEINS:
    x_scalar, x_vector = pyg_data.x
    print(f"Node Scalar Tensor Shape: {x_scalar.shape}")  # e.g., [309, 332]
    print(f"Node Vector Tensor Shape: {x_vector.shape}")  # e.g., [309, 3, 3]

    # Inspect Edge Features Tuple
    e_scalar, e_vector = pyg_data.edge_attr
    print(f"Edge Scalar Tensor Shape: {e_scalar.shape}")  # e.g., [620, 32]
    print(f"Edge Vector Tensor Shape: {e_vector.shape}")  # e.g., [620, 1, 3]

    # Inspect Edge Index Tensor (Standard PyG COO Matrix)
    print(f"Edge Index Shape: {pyg_data.edge_index.shape}")  # e.g., [2, 620] YIELDS:
    Node Scalar Tensor Shape: torch.Size([number-residues, 25])
    Node Vector Tensor Shape: torch.Size([number-residues, 3, 3])
    Edge Scalar Tensor Shape: torch.Size([number-edges (this will change depending on angstrom cutoff), 32]); 32 scalar features
    Edge Vector Tensor Shape: torch.Size([number-edges, 1, 3]) number-edges, 1 vector per edge of x,y,z

    LIGANDS: 
    print(pyg_data.x.shape)
    print(pyg_data.edge_attr.shape)
    """ 

# cast node + edge scalar/vec features into a PYG Graph. This is necessary as all the NaNs will be chopped off
import torch
import torch_geometric as pyg
import sys

def construct_graph(nodes, edges):
    # always in tuple form of (scalar, vector), even when not in 3D.
    node_scalar = torch.tensor(nodes[0], dtype=torch.float32)
    node_vector = torch.tensor(nodes[1], dtype=torch.float32)
    n_residues = node_scalar.shape[0]
    
    edge_index = torch.tensor([[i, j] for i in range(n_residues) for j in range(n_residues)], dtype=torch.long).t().contiguous()

    edge_scalar = torch.tensor(edges[0], dtype=torch.float32).flatten(0,1)
    edge_vector = torch.tensor(edges[1], dtype=torch.float32)

    is_dummy = (edge_vector.dim() == 1)
    # IF VECTOR FEATURES ARE POPULATED
    if not is_dummy:
        edge_vector = edge_vector.flatten(0,1)

    keep_indices = (~torch.isnan(edge_scalar).all(axis=-1))
    edge_index = edge_index[:,keep_indices]
    edge_scalar = edge_scalar[keep_indices,:]

    # IF VECTOR FEATURES ARE POPULATED
    if not is_dummy:
        edge_vector = edge_vector[keep_indices,:]

    edge_attr = (edge_scalar, edge_vector)

    
    pyg_data = pyg.data.Data(
        x=node_scalar,           
        x_vec=node_vector,       
        edge_index=edge_index, 
        edge_attr=edge_scalar,   
        edge_vec=edge_vector, 
        num_nodes=n_residues
    )
    # pyg_data = pyg.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_residues)
    
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

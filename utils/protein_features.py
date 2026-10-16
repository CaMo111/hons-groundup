import numpy as np
import torch
import scipy.spatial.distance as sp_dist
import ipdb


def normalise(vecs, axis=-1, eps=1e-8):
    norms = np.linalg.norm(vecs, axis=axis, keepdims=True)
    return np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms!=0)

def compute_residue_node_features(res_coords, res_idents, protein):
    # Dihedral angles (phi, psi, omega)
    dihedral_coords = res_coords[:,:3,:]
    dihedral_coords = dihedral_coords.reshape(-1, 3) # Now N, CA, C, N, CA, C, ...

    shifted_dist = dihedral_coords[1:,:] - dihedral_coords[:-1,:]
    shifted_dist = normalise(shifted_dist)

    # Backbone relative coordinates
    u0 = shifted_dist[2:,:]
    u1 = shifted_dist[1:-1,:]
    u2 = shifted_dist[:-2,:]

    # Backbone normals (cross products)
    n1 = normalise(np.cross(u1, u0, axis=-1), axis=-1)
    n2 = normalise(np.cross(u2, u1, axis=-1), axis=-1)

    # angles between normals: produces vector that is in order phi, psi, omega for each residue
    # first phi and last psi/omega is not included, which gives (3*n_residues)-3 total angles
    cosAngles = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
    allAngles = np.arccos(cosAngles) * np.sign(np.sum(n1 * u2, axis=-1))

    allAngles = np.pad(allAngles, [1,2], 'constant', constant_values=0.0)
    allAngles = allAngles.reshape(-1, 3)

    #as per pocket miner: [cos(phi), cos(psi), cos(omega),sin(phi), sin(psi), sin(omega])
    residueAngleFeats = np.concatenate([np.cos(allAngles), np.sin(allAngles)], axis=-1)
    #print('residue feats', residueAngleFeats)

    # ORIENTATION FEATURES
    alpha_coords = res_coords[:, 1, :]
    fwd = normalise(alpha_coords[1:,:] - alpha_coords[:-1,:])
    bwd = -1 * fwd
    #print('fwd', fwd)

    residueForwardFeatures = np.pad(fwd, [(0,1), (0,0)], 'constant', constant_values=0.0)
    residueBackwardFeatures = np.pad(bwd, [(1,0), (0,0)], 'constant', constant_values=0.0) 

    #print('fwd feats', residueForwardFeatures)
    
    ## Sidechain vecs
    ## sidechain is a unique cehmical group attached to AC of backbone.
    # we have N, CA, C so we extract N ++ C
    N_coords, C_coords = res_coords[:,0,:], res_coords[:,2,:]
    N_coords = normalise(N_coords - alpha_coords)
    C_coords = normalise(C_coords - alpha_coords)
    bisec = normalise(N_coords+C_coords)
    perp = normalise(np.cross(C_coords,N_coords))
    # three vals per residue for cartesian coordinates
    residueSidechainFeats = -bisec * np.sqrt(1/3) - perp * np.sqrt(2/3)
    #print('sidechain features', residueSidechainFeats) 

    ### PHYSCHEM RESIDUE LEVEL FEATURES 
    residue_letters = [protein.PROTEIN_INT_1LETTER_MAP[res] for res in res_idents]
    feature_dicts = [
            protein.AA_WEIGHTS,
            protein.AA_PKAS, 
            protein.AA_PKBS, 
            protein.AA_PKCS,
            protein.AA_PKIS,
            protein.AA_HYDROPHOB,
            protein.AA_ALIPHATIC,
            protein.AA_AROMATIC,
            protein.AA_ACIDIC, 
            protein.AA_BASIC,
            protein.AA_POLAR_NEUTRAL]

    residueAAFeats = [[d[res] for d in feature_dicts] for res in residue_letters]
    residueAAFeats = np.asarray(residueAAFeats, dtype=np.float32)
    #print('residue double A features', residueAAFeats)
    
    # FEATURE TO EXTRACT RELATIVE INDICES DIFFERENCES
    res_indexs = np.arange(len(res_idents))
    residuePosEncFeats = calc_pos_encoding(res_indexs, n_embeds=8)
    #print('positional encoding features:', residuePosEncFeats)

    ### FOR NOW WE SKIP ESM2 EMBEDDINGS
    ### COLLATE FEATURES:
    # np concat => joins arrays along an existing axis (ie same num of dimensions)
    # np.stack joins array along a new axis (ie num of dim ++)
    scalarFeatures = np.concatenate([residueAngleFeats, residueAAFeats, residuePosEncFeats], axis=-1)
    vectorFeatures = np.stack([residueForwardFeatures, residueBackwardFeatures, residueSidechainFeats], axis=1)
    nodeFeatures=(scalarFeatures, vectorFeatures)
    #print(nodeFeatures)

    return nodeFeatures

def compute_residue_edge_features(res_coords, res_idents, protein):
    # only get alpha atom coords 
    alpha_coords = res_coords[:, 1, :]
    #print('alpha coords', alpha_coords)
    #print(len(alpha_coords))

    # creates a matrix of euclidian distance from each residue to another, then flattens this into a 1D array
    edgeDist = sp_dist.squareform(sp_dist.pdist(alpha_coords))
    edgeDist = np.expand_dims(edgeDist, axis=-1)
    # IE edgeDist[32][332] --> distance between residue 33 <-> residue 333
    #print('edge dist; ', edgeDist)
    #print(edgeDist[386][1])


    # idrk what this is doing i just copied it dta-caster? from Gaussian radial basis function of this distance as a feature
    # where the distance between the means represents sqrt(2) * sigma in the RBF equation
    D_min, D_max, D_count = 0., 20., 16
    D_step = (D_max - D_min) / D_count
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = np.reshape(D_mu, [1, 1, -1])
    edgeRBFFeats = np.exp(-np.square((edgeDist - D_mu) / (D_step)))

    # creates vectors for the edges between carbon residues. Ie edgedirvec[0][1][x,y,z] the coefficients of vector between res 0 and res1
    edge_direction_vector_feats = normalise(alpha_coords[:,np.newaxis] - alpha_coords[np.newaxis,:], axis=-1)
    #print('edgedir', edge_direction_vector)
    #print(len(edge_direction_vector), len(edge_direction_vector[0]), len(edge_direction_vector[0][0]))

    #print(res_indexs)
    # FEATURE TO EXTRACT RELATIVE INDICES DIFFERENCES -> cast to length 16 vector
    # investigate this further later
    res_indexs = np.arange(len(res_idents))
    res_index_diffs = res_indexs[np.newaxis, :] - res_indexs[:, np.newaxis]
    edgePosEncFeats = calc_pos_encoding(res_index_diffs, n_embeds=16)
    #edge features before 
    edgeFeatures = np.concatenate([edgeRBFFeats, edgePosEncFeats, edge_direction_vector_feats], axis=-1)
    
    # remove all diagonal (ie self <-> info); if we NaN it out it will drop later.
    inds = np.arange(protein.protein_frame.n_residues)
    edgeFeatures[inds, inds, :] = np.nan
    edgeDist[inds, inds, :] = np.nan

    # DISTANCE MASK:
    # we take a distance approach here, ie only take within 3-4 Ang of residues/nodes:
    """"from caster dta: we instead connect each residue to every residue within a 4 Angstrom distance in 3D space, 
    with the distance threshold determined through an ablation study. The edges include self-loops of 
    a residue to itself to allow for passing of information at one node to itself in each convolution."""

    edgeDistMask = (edgeDist <= 4).astype(np.float32) # WITHIN 4 ANGSTROM: RETAINS 0.5 % of node edges 
    edgeDistMask[edgeDistMask == 0] = np.nan
    
    # NaN out the features for edges that are not within the threshold
    edgeMaskedFeatures = edgeFeatures * edgeDistMask    
    edgeScalarFeatures = edgeMaskedFeatures[:,:,:-3]    
    edgeVectorFeatures = edgeMaskedFeatures[:,:,-3:]
    edgeVectorFeatures = np.expand_dims(edgeVectorFeatures, axis=-2)   
    edgeMaskedFeatures = (edgeScalarFeatures, edgeVectorFeatures)
    return edgeMaskedFeatures


def calc_pos_encoding(indices, n_embeds=16):
    """
    Posititionally encode indices (or any values) 
    using a cosine and sine function.
    We use n_embeds/2 sin and n_embeds/2 cos embeddings, 
    with frequencies defined by the embedding number
    much like PositionalEncodings from PocketMiner 
    (and the original Attention is All You Need paper)
    """

    per_sincos = n_embeds // 2

    # log-transformed version of the
    # canonical equation used in transformer-based models
    enc_freqs = np.exp(2 * np.arange(per_sincos) * -(np.log(10000.0) / per_sincos))
    enc_expand = tuple(np.arange(len(indices.shape)))
    indices_freqs = np.expand_dims(indices, -1) * np.expand_dims(enc_freqs, enc_expand)

    return np.concatenate([np.cos(indices_freqs), np.sin(indices_freqs)], axis=-1)

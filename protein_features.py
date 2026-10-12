import numpy as np
import torch
import scipy.spatial.distance as sp_dist
import ipdb

def normalise(vecs, axis=-1, eps=1e-8):
    norms = np.linalg.norm(vecs, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return vecs / norms

def compute_residue_node_features(res_coords, res_idents):
    # Dihedral angles (phi, psi, omega)
    dihedral_coords = res_coords[:,:3,:]
    dihedral_coords = dihedral_coords.reshape(-1, 3) # Now N, CA, C, N, CA, C, ...

    shifted_dist = dihedral_coords[1:,:] - dihedral_coords[:-1,:]
    shifted_dist = normalise(shifted_dist)
    #print('shifteddistance2', shifted_dist)

    # Backbone relative coordinates
    u0 = shifted_dist[2:,:]
    u1 = shifted_dist[1:-1,:]
    u2 = shifted_dist[:-2,:]

    # Backbone normals (cross products)
    n1 = normalise(np.cross(u1, u0, axis=-1), axis=-1)
    n2 = normalise(np.cross(u2, u1, axis=-1), axis=-1)
    #print('normal vectors', n1,n2)

    # angles between normals: produces vector that is in order phi, psi, omega for each residue
    # first phi and last psi/omega is not included, which gives (3*n_residues)-3 total angles
    cosAngles = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
    allAngles = np.arccos(cosAngles) * np.sign(np.sum(n1 * u2, axis=-1))
    #print('ANGLES=', allAngles)
    #print(len(allAngles))
    allAngles = np.pad(allAngles, [1,2], 'constant', constant_values=0.0)
    allAngles = allAngles.reshape(-1, 3)
    #print(allAngles)

    #as per pocket miner: [cos(phi), cos(psi), cos(omega),sin(phi), sin(psi), sin(omega])
    residueAngleFeats = np.concatenate([np.cos(allAngles), np.sin(allAngles)], axis=-1)
    print('residue feats', residueAngleFeats)

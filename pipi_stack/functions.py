import numpy as np
from .aromatic_ring import Aromatic_Ring as AR
import copy

def calc_dist(ar1, ar2):
    """Returns the distance between two aromatic rings."""
    dist = np.linalg.norm(ar1.centroid - ar2.centroid)
    return dist

def calc_ang(ar1, ar2):
    """Returns the angle in ° between two aromatic rings."""
    ang = np.rad2deg(np.arccos(np.dot(ar1.normal, ar2.normal)))
    return ang

def calc_latdisp(ar1, ar2):
    """Returns the lateral displacement between two aromatic rings."""
    disp_vec = ar1.centroid - ar2.centroid
    
    proj_len_1 = np.abs(np.dot(disp_vec, ar1.normal))
    proj_len_2 = np.abs(np.dot(disp_vec, ar2.normal))
    
    square_dist = np.square( np.linalg.norm(disp_vec) )
    
    latdisp_1 = np.sqrt(square_dist - np.square(proj_len_1))
    latdisp_2 = np.sqrt(square_dist - np.square(proj_len_2))

    # The smallest lateral displacement of the two
    latdisp = min(latdisp_1, latdisp_2)
    return latdisp

# (Not used for any analysis...)
def calc_height(ar1, ar2):
    """Returns the height (i.e. the square root of the distance squared minus the lateral displacement squared) between two aromatic rings."""
    r = ar2.centroid - ar1.centroid
    
    h1 = np.abs(np.dot(ar1.normal,r))
    h2 = np.abs(np.dot(ar2.normal,r))
    
    hmax = max(h1,h2)
    hmin = min(h1,h2)

    return hmax, hmin

def get_boundary_rings(ar_list, box_dim, cutoff):
    """Returns a list of aromatic ring representing the periodic copies (works only for orthorhombic unit cells)"""

    # Function for copying and changing the centroid of an aromatic ring
    def make_periodic_copy(ar, new_centroid):
        ar_new = copy.copy(ar)
        ar_new.centroid = np.array(new_centroid)
        return ar_new
    
    # Check box dimensions
    a, b, c, alpha, beta, gamma = box_dim
    if [round(alpha), round(beta), round(gamma)] != [90, 90, 90]:
        raise ValueError("Peridic boundary stacking is not of yet supported for non-orthorhombic unit cells.")

    N_rings = len(ar_list) #number of rings

    # New list for periodic ring copies
    periodic_rings = []

    # Loop through each ring, check if it is close to any boundary/boundaries and if so add its periodic copy/copies to the list
    for i in range(N_rings):
        # Identifiers for periodic boundary 
        jx = 0
        jy = 0
        jz = 0

        # Load in the coordinates of the aromatic ring
        ar = ar_list[i]
        coord = ar.centroid
        x = coord[0]
        y = coord[1]
        z = coord[2]
        
        # Check in x
        if x > a - cutoff:
            x_new = x - a   
            jx += 1
        elif x < cutoff:
            x_new = x + a
            jx += 1

        # Check in y
        if y > b - cutoff:
            y_new = y - b
            jy += 1
        elif y < cutoff:
            y_new = y + b
            jy += 1

        # Check in z
        if z > c - cutoff:
            z_new = z - c
            jz += 1
        elif z < cutoff:
            z_new = z + c
            jz += 1
        
        # Add periodic copies to coordinate array if close to any boundary/boundaries
        if jx == 1: 
            ar_new = make_periodic_copy(ar, [x_new, y, z])
            periodic_rings.append(ar_new)
        if jy == 1:
            ar_new = make_periodic_copy(ar, [x, y_new, z])
            periodic_rings.append(ar_new)
        if jz == 1:
            ar_new = make_periodic_copy(ar, [x, y, z_new])
            periodic_rings.append(ar_new)
        if jx == 1 and jy  == 1:
            ar_new = make_periodic_copy(ar, [x_new, y_new, z])
            periodic_rings.append(ar_new)
        if jx == 1 and jz == 1:
            ar_new = make_periodic_copy(ar, [x_new, y, z_new])
            periodic_rings.append(ar_new)
        if jy == 1 and jz == 1:
            ar_new = make_periodic_copy(ar, [x, y_new, z_new])
            periodic_rings.append(ar_new)
        if jx == 1 and jy == 1 and jz == 1:
            ar_new = make_periodic_copy(ar, [x_new, y_new, z_new])
            periodic_rings.append(ar_new)

    return periodic_rings
import MDAnalysis as mda
import numpy as np
import scipy as sp
from .aromatic_ring import Aromatic_Ring as AR
from .functions import *
import time

def pair_cutoff(universe, idx_rings, cutoff=10, pbc=True, verbose=False):
    """
    Returns all lists of the distance, angle and lateral displacement of ring pairs whose centroids are within a cut-off distance to each other.
    
    Parameters
    ----------
    universe : MDAnalysis.universe
        A prepared trajectory from which to do the calculations.
    idx_rings : list
        A list consisting of vectors, each with the indices of the atoms of the ring.
    cutoff : float
        Cut-off distance.
    pbc : boolean  
        If true, checks pair distances across the boundary (for simulations with periodic boundary conditions). Currently only supports orthorhombic simulation boxes.
    verbose : boolean
        If true, prints progress of the calculations.

    Returns
    -------
    dist_list : list
        Contains 1D np.arrays of all distances of each ring pair for each frame.
    ang_list : list
        Contains 1D np.arrays of all angles of each ring pair for each frame.
    latdisp_list : list
        Contains 1D np.arrays of all lateral displacements of each ring pair for each frame.
    AR_pair_list : list
        Contains lists containing the two aromatic ring objects of the ring pairs for each frame.
    """

    N_frames = len(universe.trajectory)
    N_rings = len(idx_rings)
    init_arr_size = int(N_rings*cutoff**2) #overestimated size for the number of pairs with the cutoff

    dist_list = [None] * N_frames
    ang_list = [None] * N_frames
    latdisp_list = [None] * N_frames
    AR_pair_list = [None] * N_frames

    # Loop through each frame of the trajectory
    for i_frame, _ in enumerate(universe.trajectory):

        # Create a list with all aromatic rings 
        ARs = [AR( universe.select_atoms("index " + " ".join(str(idx) for idx in idx_ring)) ) for idx_ring in idx_rings]

        # Output arrays for the frame (preallocate size)
        dists = np.zeros(init_arr_size)
        angs = np.zeros(init_arr_size)
        latdisps = np.zeros(init_arr_size)
        AR_pair = [None] * init_arr_size
        i_arr = 0 #index for array

        # Add periodic copies of rings close to the boundary
        if pbc:
            box = universe.dimensions #box dimensions used for boundary stacking

            box_min_dimension = min(box[0], box[1], box[2])
            if box_min_dimension/2 < cutoff:
                raise ValueError(f"Cut-off distance is larger than half the box size {box_min_dimension/2:.2f}.")

            ARs_periodic = get_boundary_rings(ARs, box, cutoff)
            ARs.append(ARs_periodic)

        # Loop over each ring
        for i in range(N_rings):
            ar1 = ARs[i]

            # Loop over other rings
            for j in range(i+1, N_rings):
                ar2 = ARs[j]

                # Calculate distance
                dist = calc_dist(ar1, ar2)

                # If it is within cutoff, save rings, distance, angle and lateral displacement
                if dist <= cutoff:

                    # Put values in the arrays
                    dists[i_arr] = dist
                    angs[i_arr] = calc_ang(ar1, ar2)
                    latdisps[i_arr] = calc_latdisp(ar1, ar2)
                    AR_pair[i_arr] = [ar1, ar2]

                    i_arr += 1

        # Cut the arrays and put them save it to the frame
        dist_list[i_frame] = dists[:i_arr]
        ang_list[i_frame] = angs[:i_arr]
        latdisp_list[i_frame] = latdisps[:i_arr]
        AR_pair_list[i_frame] = AR_pair[:i_arr]

    return dist_list, ang_list, latdisp_list, AR_pair_list

def calculate_stackings(lists, d_range=(3.2, 4.6), ang_range=(10,50), latdisp_range=(0,2), circular=True, verbose=True):
    """
    Function for calculating pi-pi stacking of aromatic ring pairs based on three geometric critera.

    Parameters
    ----------
    lists : list
        List containing the lists of distance, angles, lateral displacement and aromatic ring pairs (same format as the outputted from the pair_cutoff function).
    d_range : tuple
        Min and max of the distance range criteria.
    ang_range : tuple
        Min and max of the angle range criteria.
    latdisp_range : tuple
        Min and max of the lateral displacement range criteria.
    circular : boolean
        Whether or not to use circular mean and standard deviation such that angles α and 180-α are the same.    
    verbose : boolean
        If True, prints the statistics.

    Returns
    -------
    dist_stack : list
        Contains 1D np.arrays of all distances of each stacked ring pair for each frame.
    ang_stack : list
        Contains 1D np.arrays of all angles of each stacked ring pair for each frame.
    latdisp_stack : list
        Contains 1D np.arrays of all lateral displacements of each stacked ring pair for each frame.
    AR_pair_stack: list
        Contains lists containing the two aromatic ring objects of the stacked ring pairs for each frame.
    """

    # Geometric criteria
    d_min, d_max = d_range
    ang_min, ang_max = ang_range
    latdisp_min, latdisp_max = latdisp_range

    ang_mid = ang_max + ang_min / 2

    # Lists
    dist_list, ang_list, latdisp_list, AR_pair_list = lists
    N_frames = len(dist_list)

    # Arrays for the stacking
    dist_stack = [None] * N_frames
    ang_stack = [None] * N_frames
    latdisp_stack = [None] * N_frames
    AR_pair_stack = [None] * N_frames
    N_stack = np.zeros(N_frames)

    # Loop over each frame
    for i_frame in range(N_frames):

        # Extract the frame data
        dist_frame = dist_list[i_frame]
        ang_frame = ang_list[i_frame]
        latdisp_frame = latdisp_list[i_frame]
        AR_pair_frame = AR_pair_list[i_frame]

        # Find indices where the conditions are met
        i = np.where((dist_frame >= d_min) & (dist_frame <= d_max))[0]
        j = np.where(((ang_frame >= ang_min) & (ang_frame <= ang_max)) | (ang_frame >= (180-ang_max)) & (ang_frame <= (180-ang_min)))[0]
        k = np.where((latdisp_frame >= latdisp_min) & (latdisp_frame <= latdisp_max))[0]
        ijk = np.intersect1d(i, np.intersect1d(j,k))

        # Extract values within conditions and save data for the frame
        dist_stack[i_frame] = dist_frame[ijk]
        ang_stack[i_frame] = ang_frame[ijk]
        latdisp_stack[i_frame] = latdisp_frame[ijk]
        AR_pair_stack[i_frame] = [AR_pair_frame[idx] for idx in ijk] 

        N_stack[i_frame] = len(ijk) #save the number of stackings for the frame
        
    # Flatten the arrays
    dist_flat = np.concatenate(dist_stack)
    ang_flat = np.concatenate(ang_stack)
    latdisp_flat = np.concatenate(latdisp_stack)

    # Calculate the statistics
    dist_mean = np.mean(dist_flat) 
    dist_std = np.std(dist_flat)
    
    if circular:
        ang_flat[ang_flat>90] = 180 - ang_flat[ang_flat>90]

    ang_mean= np.mean(ang_flat) 
    ang_std = np.std(ang_flat)
    
    latdisp_mean = np.mean(latdisp_flat)
    latdisp_std = np.std(latdisp_flat)
    
    N_mean = np.mean(N_stack)
    N_std =  np.std(N_stack)

    # Print the statistics
    if verbose:
        print("---Average values of the stackings---")
        print(f"Number per frame: {N_mean:.0f} +/- {N_std:.0f}")
        print(f"Distance: {dist_mean:.2f} +/- {dist_std:.2f}")
        print(f"Angle: {ang_mean:.1f} +/- {ang_std:.1f}")
        print(f"Lateral displacement: {latdisp_mean:.2f} +/- {latdisp_std:.2f}")

    return dist_stack, ang_stack, latdisp_stack, AR_pair_stack

def criteria_2D(A_list, A_range, B_list, B_range, C_list):
    """
    Takes the indices of two lists A and B based on value range conditions for each list and returns the corresponding values in the list C.
    The lists should correspond to the distance, angle and lateral displacement of aromatic ring pairs.

    Parameters
    ----------
    A_list : list
        Data of list A for each frame.
    A_range : tuple
        The range condition for the elements of list A on the form (min, max).        
    B_list : list
        Data of list B for each frame.
    B_range : tuple
        The range condition for the elements of list B on the form (min, max).
    C_list : list
        Data of list C for each frame.

    Returns
    -------
    C_values : list
        Flat list containing all the values for list C based on the conditions of the other two lists.
    """

    # Check that the lists are same size
    if not (len(A_list) == len(B_list) == len(C_list)) or not all(len(A_list[i]) == len(B_list[i]) == len(C_list[i]) for i in range(len(A_list))):
        raise ValueError("Lists are of different shapes.")
    
    # Flatten the lists
    A_flat = np.concatenate(A_list)
    B_flat = np.concatenate(B_list)
    C_flat = np.concatenate(C_list)

    # Find the indices based on A and B conditions
    i = np.where((A_flat >= A_range[0]) & (A_flat <= A_range[1]))[0]
    j = np.where((B_flat >= B_range[0]) & (B_flat <= B_range[1]))[0]
    ij = np.intersect1d(i,j)

    # Extract values for C
    C_values = C_flat[ij]

    return C_values

def criteria_1D(A_list, A_range, B_list, C_list):
    """
    Takes the indices of two lists A and B based on value range conditions for each list and returns the corresponding values in the list C.
    The lists should correspond to the distance, angle and lateral displacement of aromatic ring pairs.

    Parameters
    ----------
    A_list : list
        Data of list A for each frame.
    A_range : tuple
        The range condition for the elements of list A on the form (min, max).        
    B_list : list
        Data of list B for each frame.
    C_list : list
        Data of list C for each frame.

    Returns
    -------
    B_values : list
        Flat list containing all the values for list B based on the conditions of list A.
    C_values : list
        Flat list containing all the values for C based on the conditions of list A.
    """

    # Check that the lists are same size
    if not (len(A_list) == len(B_list) == len(C_list)) or not all(len(A_list[i]) == len(B_list[i]) == len(C_list[i]) for i in range(len(A_list))):
        raise ValueError("Lists are of different shapes.")
    
    # Flatten the lists
    A_flat = np.concatenate(A_list)
    B_flat = np.concatenate(B_list)
    C_flat = np.concatenate(C_list)

    # Find the indices based on A and B conditions
    i = np.where((A_flat >= A_range[0]) & (A_flat <= A_range[1]))[0]
    
    # Extract values for B and C
    B_values = B_flat[i]
    C_values = C_flat[i]

    return B_values, C_values
    





        







import numpy as np
import os

def calc_distance(ring_i_COM, ring_j_COM, box_dim):
    ring_i_COM = np.asarray(ring_i_COM, dtype=np.float64)
    ring_j_COM = np.asarray(ring_j_COM, dtype=np.float64)
    box_dim = np.asarray(box_dim, dtype=np.float64)
    delta = ring_i_COM - ring_j_COM
    delta -= np.round(delta / box_dim) * box_dim
    return np.linalg.norm(delta)

# gives the ring coordinates
def extractRingCoordsfromPDB(pdbname,N_molecules,length_molecules,i_rings):
    nr_atoms = N_molecules*len(i_rings)*len(i_rings[0])
    coords = np.zeros([nr_atoms,3])
    target_rows = [item for sublist in i_rings for item in sublist]
    i = 0
    j = 0
    with open(pdbname,'r') as file:
        for line in file:
            entries = line.split()
            if len(entries) >= 1:
                record = entries[0]
                if record == 'ATOM':
                    if j>nr_atoms: print('you probably forgot to remove the solvent from the frame.pdb files')
                    i += 1
                    row_pos = i % length_molecules
                    if row_pos in target_rows:
                        x = float(line[31:38])
                        y = float(line[39:46])
                        z = float(line[47:54])
                        coords[j,:] = [x,y,z]
                        j += 1

                elif record == 'CRYST1':
                    box_dim = np.array([float(entries[1]),float(entries[2]),float(entries[3])])
    return coords, box_dim

# calculates center of mass and surface normal of rings
def calcCOMandNormal(coords):
    N_rings = int(len(coords)/6)
    COM_coords = np.zeros([N_rings,3])
    normal_vector = np.zeros([N_rings,3])
    for i in range(N_rings):
        i1 = 6*i
        i2 = 6*(i+1)
        points = coords[i1:i2,:]
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        _, _, vh = np.linalg.svd(points_centered)
        normal = vh[-1]
        normal_vector[i] = normal
        COM_coords[i] = centroid  
    return COM_coords, normal_vector

# returns the lateral displacement between the rings (smallest of the two)
# edited to take the pbc into account
def calcLateralDisplacement(r1,r2,n1,n2,box_dim):
    displacement_vector = r1 - r2
    displacement_vector -= box_dim * np.round(displacement_vector / box_dim)

    # Compute proper squared distance after periodic correction
    square_dist = np.square(np.linalg.norm(displacement_vector))

    # Normalize n1 and n2 to ensure correct projections
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Compute projections
    projection_length_i = np.abs(np.dot(displacement_vector, n1))
    projection_length_j = np.abs(np.dot(displacement_vector, n2))

    # Ensure non-negative values inside sqrt
    lat_disp_i = np.sqrt(max(0, square_dist - np.square(projection_length_i)))
    lat_disp_j = np.sqrt(max(0, square_dist - np.square(projection_length_j)))

    return min(lat_disp_i, lat_disp_j)

# returns stacking parameters for pairs of d < dmax
def calcDistrib(framename,name_molecule,lignin_type):
    # Info about the lignin molecules
    list_nr_atom = {'name_molecule':nr of atoms}
    #eg.: list_nr_atom = {'Gbetao4G':51,'Gbetao4Gbetao4Gbetao4G': 103}

    list_nr_mol = {'type molecule':nr of lignin molecules in the box}
    #eg.: list_nr_mol = {'dimer':200,'tetramer':100}

    list_nr_ring = {'type molecule':nr of rings in one molecule}
    #eg.: list_nr_ring = {'dimer':2, 'tetramer':4}

    def get_atomnumber(name_molecule,N_ring):
        if name_molecule == "name molecule":
            list_atomnumbers = {"nr ring":[C1,C2,C3,C4,C5,C6],...}
        #eg.:if name_molecule=='Gbetao4G':
        #    list_atomnumbers={1:[1, 2, 4, 6, 5, 3],2:[15, 14, 12, 10, 11, 13]}

    N_rings = list_nr_ring.get(lignin_type)
    i_rings = []
    for i in range(1,N_rings+1,1):
        r_i=get_atomnumber(name_molecule,i)
        i_rings.append(r_i)
    N_molecules = list_nr_mol.get(lignin_type)
    N_atom = list_nr_atom.get(name_molecule)
    d_max = 10
    index = 0

    # output arrays
    distance = np.zeros(N_molecules*N_rings*N_atom)
    angle = np.zeros(N_molecules*N_rings*N_atom)
    lateral_displacement = np.zeros(N_molecules*N_rings*N_atom)
    ring_index_pairs = np.zeros((N_molecules*N_rings*N_atom,2))
    
    length_molecules=list_nr_atom.get(name_molecule)
    coords, box_dim = extractRingCoordsfromPDB(framename,N_molecules,length_molecules,i_rings)

    COM_rings, NV_rings = calcCOMandNormal(coords)
    
  
    # Loop over each ring
    for i in range(len(COM_rings)):
        ring_i_COM = COM_rings[i,:]
        ring_i_NV = NV_rings[i,:]  
            
        # Loop over other rings
        for j in range(i+1,len(COM_rings)):
            ring_j_COM = COM_rings[j,:]
            ring_j_NV = NV_rings[j,:]
                
            # calculate distance
            d = calc_distance(ring_i_COM,ring_j_COM,box_dim)

            # SANDWICH
            if d < d_max:
                distance[index] = d
                angle[index] = np.arccos(np.dot(ring_i_NV,ring_j_NV))
                lateral_displacement[index] = calcLateralDisplacement(ring_i_COM,ring_j_COM,ring_i_NV,ring_j_NV,box_dim)
                ring_index_pairs[index,0] = i+1
                ring_index_pairs[index,1] = j+1
                index += 1
    
    distance = distance[0:index]
    angle = angle[0:index]
    lateral_displacement = lateral_displacement[0:index]
    ring_index_pairs = ring_index_pairs[0:index,:]

    return distance, angle, lateral_displacement, ring_index_pairs

# main()
import sys

# Read input arguments
name_molecule = sys.argv[1]

frame_folder = "frames/"

list_lignin_type={'name molecule':'type molecule'}
#eg.: list_lignin_type={'Gbetao4G':'dimer','Gbetao4Gbetao4Gbetao4G':'tetramer'}

lignin_type = list_lignin_type.get(name_molecule)
frames = os.listdir(frame_folder)

distances = []
angles = []
lateral_displacements = []
ring_index_pairs = []

# print(str(len(frames)) + " frames from: " + folder)
# print("lignin type: ", lignin_type)
for frame in frames:       
    print("calculating stacking for",frame,end='')   
        
    d, ang, lat_disp, ri_pairs = calcDistrib(frame_folder+frame,name_molecule,lignin_type)

    distances.append(d)
    angles.append(ang)
    lateral_displacements.append(lat_disp)
    ring_index_pairs.append(ri_pairs)
        
    print("...done",end='\n')

distances = np.array(distances, dtype=object)
angles = np.array(angles, dtype=object)
lateral_displacements = np.array(lateral_displacements, dtype=object)
ring_index_pairs = np.array(ring_index_pairs, dtype=object)

np.save("distances.npy",distances,allow_pickle=True)
np.save("angles.npy",angles,allow_pickle=True)
np.save("lateral_displacements.npy",lateral_displacements,allow_pickle=True)
np.save("ring_index_pairs.npy",ring_index_pairs,allow_pickle=True)

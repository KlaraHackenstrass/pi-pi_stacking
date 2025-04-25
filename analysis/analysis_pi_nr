# analysis_pi_nr.py:
import sys
import numpy as np

# Read input arguments
name_molecule = sys.argv[1]     #same as in folder name
stacking_type = sys.argv[2]     #sandwich or T-shaped

print('working on: ')
print(name_molecule+' '+stacking_type)

    list_lignin_type={'name molecule':'type molecule'}
    #eg.: list_lignin_type={'Gbetao4G':'dimer','Gbetao4Gbetao4Gbetao4G':'tetramer'}

    list_nr_ring = {'type molecule':nr of rings in one molecule}
    #eg.: list_nr_ring = {'dimer':2, 'tetramer':4}

# data location
lignin_type = list_lignin_type.get(name_molecule)
nr_rings = list_nr_ring.get(lignin_type)

# load the data raw
distance_raw = np.load("distances.npy",allow_pickle=True)
angle_raw = np.load("angles.npy",allow_pickle=True)
lateral_displacement_raw = np.load("lateral_displacements.npy",allow_pickle=True)
ring_index_pairs_raw = np.load("ring_index_pairs.npy",allow_pickle=True)

N_frames = len(distance_raw)    # number of frames

if stacking_type == 'sandwich':
    d = 3.9
    d_tol = 0.7
    ang = 30
    ang_tol = 20
    l = 1.0
    l_tol = 1.0
elif stacking_type == 'T-shape':
    d = 5.0 
    d_tol = 1.0
    ang = 80
    ang_tol = 10
    l = 0.9
    l_tol = 0.9
    
list_intra, list_inter = [],[]
lat_disp_inter,lat_disp_intra = [],[]
angle_inter,angle_intra  = [],[]
distance_inter,distance_intra = [],[]
    
for i in range(N_frames):
    counter_inter, counter_intra = 0,0
    # accessing the data that was compiled using "pre-conditions"
    distance_all = distance_raw[i]
    ang_0to180 = angle_raw[i]*180/np.pi
    angle_all = np.where(ang_0to180 > 90, 180 - ang_0to180, ang_0to180)   #correct to angles between 0 and 90
    lat_disp_all = lateral_displacement_raw[i]
    ri_pairs = ring_index_pairs_raw[i]

    #condition 1: distance
    indices_dist = np.where((distance_all >= d - d_tol) & (distance_all <= d + d_tol))[0]
    #condition 2: angles
    indices_ang = np.where((angle_all >= ang - ang_tol) & (angle_all <= ang + ang_tol))[0]
    #condition 3: lateral displacement
    indices_latdisp = np.where((lat_disp_all >= l - l_tol) & (lat_disp_all <= l + l_tol))[0]

    #find the indices that fulfill all 3 conditions
    indices_dal = np.intersect1d(indices_dist,np.intersect1d(indices_ang,indices_latdisp))

    for index in indices_dal:
        ri_pair = ri_pairs[index]
        i1 = ri_pair[0]     #index of ring1
        i2 = ri_pair[1]     #index of ring2
    
        m1 = (i1-1) // nr_rings #molecule nr of ring1
        m2 = (i2-1) // nr_rings #molecule nr of ring2

        if m1 != m2:
            counter_inter += 1
            lat_disp_inter.append(lat_disp_all[index])
            angle_inter.append(angle_all[index])
            distance_inter.append(distance_all[index])

        else:
            counter_intra += 1
            lat_disp_intra.append(lat_disp_all[index])
            angle_intra.append(angle_all[index])
            distance_intra.append(distance_all[index])
        
    list_intra.append(counter_intra)
    list_inter.append(counter_inter)


N_intra_avg = np.average(list_intra)
N_intra_std = np.std(list_intra)
N_inter_avg = np.average(list_inter)
N_inter_std = np.std(list_inter)

with open('pi_pi_nr_{}.txt'.format(stacking_type), "w") as f:
    print('----------INTRA----------',file=f)
    print(N_intra_avg,' +/- ',N_intra_std,file=f)
    
    print('----------INTER----------',file=f)
    print(N_inter_avg,' +/- ',N_inter_std,file=f)
    f.close()

np.savetxt(lat_dist_inter_{}.txt'.format(stacking_type),lat_disp_inter)
np.savetxt('lat_dist_intra_{}.txt'.format(stacking_type),lat_disp_intra)
np.savetxt('ang_inter_{}.txt'.format(stacking_type),angle_inter)
np.savetxt('ang_intra_{}.txt'.format(stacking_type),angle_intra)
np.savetxt('dist_inter_{}.txt'.format(stacking_type),distance_inter)
np.savetxt('dist_intra_{}.txt'.format(stacking_type),distance_intra)

\end{verbatim}

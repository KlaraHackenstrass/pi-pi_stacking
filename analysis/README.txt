Approach to calculate $\pi-\pi$ stacking

step 1: Generation of pdb files of each frame
echo "LIG" | srun -n 1 gmx trjconv -f prod.xtc -s prod.tpr -o frames/conf_.pdb -sep -pbc mol

step 2: Prepare the data
This script reads the frames and generates .npy files for the distance, angle, lateral displacement and index files each time two rings are within 100 Å. This is the most time consuming step, so it is separated from the rest of the analysis to allow more flexible analysis that does not require repeating the analysis if for example cut-off values are changed.
Run this command as: 
    python3 -u prep_pi.py "name_molecule"
    eg. python3 -u prep_pi.py "Gbetao4G"


step 3: Calculate the nr of stackings
This scripts calculates the number of intra and intermolecular stackings either using the T-shaped or sandwich criteria. Run this script as:
python3 -u analysis_pi_nr.py "name moleucle" "type of stacking: sandwich or T-shape"
example
python3 -u analysis_pi_nr.py "Gbetao4G" T-shape

# Project
Updated version of the python code for calculating pi-pi stacking of aromatic rings from molecular dynamics trajectories of lignin. The old code and data are in the direcotry "old_data/". 

## Notes on updated version
Added a class for the aromatic rings, changed the code to a package and added numpy docstrings and ValueError:s to the functions. Also incorporated MDAnalysis to load and treat the trajectories rather than going in to the coordinate file and extracting all information. This makes the code applicable to other systems than the ones tested in the article. The example.ipynb notebook goes over an example, which should be enough for a new user to use the code.    


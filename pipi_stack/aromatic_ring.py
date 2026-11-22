import MDAnalysis as mda
import numpy as np

class Aromatic_Ring:
    """A class for creating aromatic rings from an atom group selection."""

    def __init__(self, atom_group):
        # Atom indices
        self.indices = atom_group.indices     

        # Centroid of the ring gives the position
        centroid = atom_group.center_of_geometry()   
        self.centroid = centroid
        
        # Normal vector
        pos = atom_group.positions
        _, _, vh = np.linalg.svd(pos-centroid)
        self.normal = vh[-1]

    def __str__(self):
        location = ",".join(str(round(i, 2)) for i in self.centroid)
        vector = ",".join(str(round(i, 2)) for i in self.normal)
        return f"Aromatic ring located at (x,y,z) = ({location}) with normal vector (nx,ny,nx) = ({vector}). "
    

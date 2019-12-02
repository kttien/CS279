from vtkplotter import *

volume = load(mydicomdir) #returns a vtkVolume object
show(volume, bg='white')
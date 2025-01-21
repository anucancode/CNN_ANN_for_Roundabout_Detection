import h5py
import numpy as np
filename = "roundabout_center_radius_model.hdf5"
import matplotlib.pyplot as plt
from tensorflow import *

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f['conv2d'])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = f['conv2d']
    
    data = image.convert_image_dtype(data, dtype=float32)
    #data = expand_dims(data, axis=0)

    plt.imshow(
    squeeze(data)
    )
'''
    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f['conv2d'])
    # preferred methods to get dataset values:
    ds_obj = f['conv2d']      # returns as a h5py dataset object
    ds_arr = f['conv2d'][()]  # returns as a numpy array
'''
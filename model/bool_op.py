import os
import h5py
import json
import numpy as np
from cadlib.visualize import vec2CADsolid
from cadlib.macro import *
from cadlib.extrude import *
from OCC.Core.BRepAlgoAPI import *

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add


def bool_op(cfg, vec, is_numerical=True):

    data = vec

    if cfg.run_inference_reverse:
        # get shape of whole shape
        
        whole_shape = vec2CADsolid(data, is_numerical=is_numerical)

    ending_positions = np.where(data[:, 0] > SOL_IDX)    #np.where(data[:, 0] == 4)

    # split using start positions
    
    split_data = np.split(data, ending_positions[0]+1)[:-1] # [:-1] remove the last empty array

    accumulated_array = []
    if cfg.run_inference_reverse:
        for i in range(len(split_data)-1):
            accumulated_array.append(np.concatenate(split_data[:len(split_data)-i-1]))
    else:
        for i in range(len(split_data)):
            accumulated_array.append(np.concatenate(split_data[:i+1]))

    if cfg.run_inference_reverse:
        cutted_shape = [whole_shape]
    else:
        cutted_shape = [None]
    for i in range(len(accumulated_array)):
        cutted_shape.append(vec2CADsolid(accumulated_array[i], is_numerical=is_numerical))
    if cfg.run_inference_reverse:
        cutted_shape.append(None)

    return cutted_shape 



# if __name__ == '__main__':
#     cutted = bool_op("00019_index_6.json")
#     # for i, shape in enumerate(cutted):
#     views_np_arrays = get_three_view_shapes(cutted)
#     # show all pil images
#     for views in views_np_arrays:
#         for view_name, view in views.items():
#             view.show()

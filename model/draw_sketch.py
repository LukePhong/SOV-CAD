import os
import h5py
import json
import numpy as np
from cadlib.visualize import vec2CADsolid
from cadlib.macro import *
from cadlib.extrude import *
from cadlib.sketch import *
from OCC.Core.BRepAlgoAPI import *

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_step_sketch_images(cfg, cad_seq, img_height=256, img_width=256, line_width=2, line_color='black'):


    seq_list = cad_seq.seq
    all_img = []
    for i, item in enumerate(seq_list):
        img = []
        if isinstance(item, Extrude) or isinstance(item, Revolve):
            sketch = item.profile
            dpi = 100
            
            # Create a figure with the exact requested dimensions
            fig = plt.figure(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
            
            # Create axis with no padding
            ax = fig.add_axes([0, 0, 1, 1])
            
            # Draw the sketch to determine bounds
            sketch.draw(ax)
            
            # Set aspect equal for proper line lengths
            ax.set_aspect('equal')
            
            # Clear for redraw
            ax.clear()
            
            # Remove all axis elements and frame AFTER clearing
            ax.axis('off')
            
            # Redraw with proper styling
            sketch.draw_image(fig, ax, img, True, line_width=line_width, line_color=line_color)
            
            # Make sure figure is rendered with all adjustments
            fig.canvas.draw()
            
            # Close the figure
            plt.close(fig)

            # for rev axis
            if isinstance(item, Revolve):
                img.append(img[-1])
            
        all_img.append(img)

    if cfg.run_inference_reverse:
        all_img = np.flip(all_img, axis=0)

    return all_img


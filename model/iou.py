from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties,brepgprop_VolumeProperties
from OCC.Core.BRepAlgoAPI import (
    BRepAlgoAPI_Fuse,
    BRepAlgoAPI_Common
)
import numpy as np


def cal_volume(shape):
    if not shape:   # shape is None
        return 0
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    volume = props.Mass()
    if volume < 0:

        return 0
    return volume

def cal_iou_and_shapes(shape1, shape2):
    if not shape1 or not shape2:
        return 0., None, None, None, None
    if not (cal_volume(shape1) and cal_volume(shape2)):
        return 0., None, None, None, None
    
    fused_shape = BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    common_shape = BRepAlgoAPI_Common(shape1, shape2).Shape()

    
    fused_volume = cal_volume(fused_shape)
    common_volume = cal_volume(common_shape)

    
    iou = common_volume / fused_volume if fused_volume != 0 else 0
    if iou > 1.:
        print("iou > 1, set iou = 1")
        iou = 1.
    return np.round(iou, 4), fused_shape, common_shape, shape1, shape2

def get_iou(shape1, shape2):
    return cal_iou_and_shapes(shape1, shape2)[0]

def get_iou_seq(cfg, shape_whole, shape_cutted):
    iou_seq = []
    if cfg.run_inference_reverse:
        iou_seq = [0.0]
    for i in range(len(shape_cutted)):
        iou, fused_shape, common_shape, shape1, shape2 = cal_iou_and_shapes(shape_whole, shape_cutted[i])
        if cfg.run_inference_reverse:
            iou_seq.append(1.0 - iou)
        else:
            iou_seq.append(iou)
    return iou_seq

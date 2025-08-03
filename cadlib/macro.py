import numpy as np
## Target dictionary elements
ALL_COMMANDS = ['Line', 'Arc', 'Circle', 'RevA', 'EOS', 'SOL', 'Ext', 'ExtCut', 'Rev', 'RevCut','Fillet','Chamfer'] ##
LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')
EXT_IDX = ALL_COMMANDS.index('Ext')
EXTCUT_IDX = ALL_COMMANDS.index('ExtCut')
REV_IDX = ALL_COMMANDS.index('Rev')
REVA_IDX = ALL_COMMANDS.index('RevA')
REVCUT_IDX = ALL_COMMANDS.index('RevCut')
FILLET_IDX = ALL_COMMANDS.index('Fillet')
CHAMFER_IDX = ALL_COMMANDS.index('Chamfer')
# PAD_IDX = ALL_COMMANDS.index('PAD')


EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]

EXTENT_TYPE = ["OneSideFeatureExtentType", "BothSidesFeatureExtentType"]

# REVOLVE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
#                       "CutFeatureOperation"]

## Count of arguments
PAD_VAL = -1
### Use one-hot representation for the following parameters
N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
N_ARGS_ROTAT = 3 # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4 # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 2 # extrusion parameters: e1, e2, ###Currently only consider e1, e2 to reduce parameter count, so don't consider the last two parameters: u, flag: flag indicating mirroring
N_ARGS_EXT = N_ARGS_ROTAT + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS_TRA = 3
N_ARGS_AXIS_PARAM = 0 # Currently can convert Axis to a line with origin at (0,0) and direction (1,0) through translation, so Axis has no extra parameters here, corresponding parameters are only N_ARGS_PLANE and N_ARGS_TRANS
N_ARGS_REV_PARAM = 1 # revolve parameters: revolve angle, because Revolve's Axis is separated out, so here the parameters for Revolve only have rotation angle
N_ARGS_FILL_PARAM = 5 # fillet or chamfer parameters: x, y, z, d1, d2
N_ARGS_PLANE = N_ARGS_ROTAT + N_ARGS_TRANS
N_ARGS = N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM

SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS)])
EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS)])
# PAD_VEC = np.array([PAD_IDX, *([PAD_VAL] * N_ARGS)])

## MUST STAY WITH THE SAME ORDER TO ALL_COMMANDS
## Two Masks are used to ignore args that don't belong to the command itself when calculating loss, and also used in logits2vec function
CMD_ARGS_MASK = np.array([
                        [1, 1, 0, 0, 0, *[0]*(N_ARGS - N_ARGS_SKETCH)],  # line
                        [1, 1, 1, 1, 0, *[0]*(N_ARGS - N_ARGS_SKETCH)],  # arc
                        [1, 1, 0, 0, 1, *[0]*(N_ARGS - N_ARGS_SKETCH)],  # circle
                        [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[0]*(N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)],  # RefAxis
                        [*[0]*N_ARGS],  # EOS
                        [*[0]*N_ARGS],  # SOL
                        [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[1]*N_ARGS_EXT_PARAM, *[0]*(N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)],  # EXT
                        [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[1]*N_ARGS_EXT_PARAM, *[0]*(N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)],  # EXT-Cut
                        [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[0]*N_ARGS_EXT_PARAM, 1, *[0]*N_ARGS_FILL_PARAM],  # Rev
                        [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[0]*N_ARGS_EXT_PARAM, 1, *[0]*N_ARGS_FILL_PARAM],  # Rev-Cut
                        [*[0]*(N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM),1, 1, 1, 1, 1],  # Fillet
                        [*[0]*(N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM),1, 1, 1, 1, 1],  # Chamfer
                        ])

## Used to specify sentence length to enable batch processing
NORM_FACTOR = 1.0 # scale factor for normalization to prevent overflow during augmentation

MAX_N_EXT = 277 # maximum number of extrusion
MAX_N_LOOPS = 60 # 6 # maximum number of loops per sketch
MAX_N_CURVES = 273 # 30 # maximum number of curves per loop
MAX_TOTAL_LEN = 1220 # 64 # maximum cad sequence length
# ARGS_DIM = 256

##Face Max Num
# FACE_MAX_NUM = 256
# Local_MAX_NUM = 1024


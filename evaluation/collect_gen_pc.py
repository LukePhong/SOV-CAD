import json
import os
import glob
import traceback
import numpy as np
import h5py
from joblib import Parallel, delayed
import argparse
import sys
# sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.config import Config
from utils.pc_utils import write_ply
from cadlib.visualize import vec2CADsolid, CADsolid2pc
from model.utils import read_step_file


parser = argparse.ArgumentParser()
# parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
args = parser.parse_args()

cfg = Config()
SAVE_DIR = os.path.join(cfg.eval_root, "test_gt_pc")
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

def process_one(path):
    data_id = path.split("/")[-1][:-3]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    print("[processing] {}".format(data_id))
    with h5py.File(path, 'r') as fp:
        out_vec = fp["vec"][:].astype(float)

    try:
        # shape = read_step_file(path)
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("read_step_file failed", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None
    
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)


# all_paths = glob.glob(os.path.join(args.src, "*.h5"))
# Load data file paths
with open(cfg.dataset_file, 'r') as f:
    dataset_file = json.load(f)
all_paths = dataset_file['test']
# full paths
all_paths = [os.path.join(cfg.gt_vec_path, x + ".h5") for x in all_paths]

batch_size = 32

batches = [all_paths[i:i + batch_size] for i in range(0, len(all_paths), batch_size)]
for batch in batches:
    try:
        Parallel(n_jobs=8, backend="multiprocessing", verbose=16, timeout=300)(delayed(process_one)(x) for x in batch)
    except Exception as e:
        print("error in batch")
        tb = traceback.format_exc()
        print(tb)
        # break
        continue
print("All done")



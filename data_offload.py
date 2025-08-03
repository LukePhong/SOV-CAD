import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import copy
from joblib import Parallel, delayed
import multiprocessing
import time
import signal
import traceback
import concurrent.futures

# Set the start method to 'spawn' to avoid daemon process limitations
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from model.config import Config
from model.bool_op import *
from model.three_view import *
from model.draw_sketch import *
from model.iou import *
from model.utils import read_step_file
from cadlib.visualize import export_step_file, create_CAD

def add_start_point(vec):
    # find where to add start points
    line_arc_pos = np.where((vec[:, 0] == LINE_IDX) | (vec[:, 0] == ARC_IDX))[0]
    if len(line_arc_pos) == 0:
        return vec
    last_skt_op_pos = np.where(np.diff(line_arc_pos) > 1)[0]
    # filter out Circles
    last_skt_op_pos = last_skt_op_pos[np.where(vec[line_arc_pos[last_skt_op_pos]+1, 0] != CIRCLE_IDX)[0]]
    if len(last_skt_op_pos) > 0:
        first_skt_op_pos = last_skt_op_pos + 1
        first_skt_op_pos = np.insert(first_skt_op_pos, 0, 0)
    else:
        first_skt_op_pos = 0
    last_skt_op_pos = np.insert(last_skt_op_pos, len(last_skt_op_pos), len(line_arc_pos)-1)     # add the last skt op
    to_add_pos = line_arc_pos[first_skt_op_pos]
    # find all the start points
    last_skt_op = vec[line_arc_pos[last_skt_op_pos]]
    # turn last_skt_op to LINE
    last_skt_op[:, 0] = LINE_IDX
    vec_mask = ~CMD_ARGS_MASK[LINE_IDX].astype(bool)
    vec_mask = np.insert(vec_mask, 0, False)    # pad a False at the beginning
    last_skt_op[:, vec_mask] = -1
    start_points = last_skt_op
    # add start points to vec
    new_vec = np.insert(vec, to_add_pos, start_points, axis=0)
    return new_vec

def get_gt_view(cfg, file, height, width, from_json=True):
    if from_json:   
        json_path = Path(cfg.json_data) / (file+".json")
        with open(json_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
        shape = create_CAD(CADSequence.from_dict(json_data), False)
    else:
        gt_path = Path(cfg.step_path) / (file+".step")
        shape = read_step_file(str(gt_path))
    gt_view = get_three_view_shapes([shape], height, width)[0]
    return gt_view

def calculate_mean_std(gt_view, shapes_view, sketch_img):
    mean = []
    std = []

    # compute mean and std for gt_view
    # gt_view is (3, height, width, 3)
    for view in gt_view:
        for i in range(3):    # 3 channels
            mean.append(np.mean(view[:, :, i]))
            std.append(np.std(view[:, :, i]))

    shapes_view = np.array(shapes_view)
    # flatten sketch_img
    sketch_img = np.array([sketch for step in sketch_img for sketch in step])
    # compute mean and std for each channel
    for views in range(shapes_view.shape[1]):   # views: top, front, side
        for i in range(3):    # 3 channels
            mean.append(np.mean(np.stack([steps[:, :, i] for steps in shapes_view[:, views]])))
            std.append(np.std(np.stack([steps[:, :, i] for steps in shapes_view[:, views]])))
    
    # compute mean and std for sketch_img
    for i in range(3):    # 3 channels
        mean.append(np.mean(np.stack([step[:, :, i] for step in sketch_img])))
        std.append(np.std(np.stack([step[:, :, i] for step in sketch_img])))
    return mean, std

def get_vec_shape(cfg, filename):
    json_path = Path(cfg.json_data) / (filename+".json") 
    with open(json_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)
    cad_seq = CADSequence.from_dict(json_data)

    # shape = create_CAD(copy.deepcopy(cad_seq), False)
    
    cad_seq.normalize()                    # normalize

    shape = create_CAD(copy.deepcopy(cad_seq), False)

    cad_seq.numericalize()                 # numericalize()
    vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)

    vec_no_start = vec

    vec = add_start_point(vec)

    op_pos_before = np.where(vec[:, 0]>SOL_IDX)[0]
    # remove SOL & EOS
    vec_raw = vec[(vec[:, 0] != SOL_IDX) & (vec[:, 0]!= EOS_IDX)]
    # cut to max_total_len
    vec_raw = vec_raw[:cfg.max_total_len-1]     #NOTE -1 for first episode
    # if there's sketch ops left at end, remove them
    op_pos = np.where(vec_raw[:, 0]>SOL_IDX)[0]
    vec_raw = vec_raw[:op_pos[-1]+1]
    # cut cad_seq.seq to the same
    cad_seq.seq = cad_seq.seq[:len(op_pos)]

    # cut vec
    vec = vec[:op_pos_before[len(op_pos)-1]+1]
    
    # Also cut vec_no_start to match the effective info length in vec
    # We need to identify which elements in vec_no_start correspond to the ones kept in vec
    
    # First, determine which operations to keep
    if len(op_pos_before) > 0 and len(op_pos) > 0:
        # Keep operations up to the last one in vec
        kept_op_pos = op_pos_before[:len(op_pos)]
        
        # Find all SOL positions in the original
        op_positions = np.where(vec_no_start[:, 0] > SOL_IDX)[0]
        
        # Find operations in the original vec_no_start that should be kept
        if len(op_positions) > 0:
            # Count operations to keep from start points
            vec_no_start = vec_no_start[:op_positions[len(kept_op_pos)-1]+1]
    
    # Split vec_raw into fragments based on operation positions
    vec_raw = np.split(vec_raw, op_pos+1)
    if cfg.run_inference_reverse:
        vec_raw = np.flip(vec_raw, axis=0)
    vec_raw = np.concatenate(vec_raw, axis=0)
    
    return vec_raw, vec, cad_seq, vec_no_start, shape

def create_data_dirs(base_path):
    """Create directory structure for different data types"""
    dirs = {
        'gt_view': Path(base_path) / 'gt_view',
        'vec_raw': Path(base_path) / 'vec_raw',
        'shapes_view': Path(base_path) / 'shapes_view',
        'sketch_img': Path(base_path) / 'sketch_img',
        'iou_seq': Path(base_path) / 'iou_seq'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)
    
    return dirs

def save_data(data_dirs, filename, gt_view, vec_raw, shapes_view, sketch_img, iou_seq):
    """Save different data types to their respective folders using H5 format with compression"""

    filename = filename.split('/')[-1]

    # Save ground truth view
    with h5py.File(data_dirs['gt_view'] / f"{filename}.h5", 'w') as h5f:
        h5f.create_dataset('data', data=gt_view, compression="lzf")
    
    # Save vector data
    with h5py.File(data_dirs['vec_raw'] / f"{filename}.h5", 'w') as h5f:
        h5f.create_dataset('data', data=vec_raw, compression="lzf")
    
    # Save shapes view
    with h5py.File(data_dirs['shapes_view'] / f"{filename}.h5", 'w') as h5f:
        h5f.create_dataset('data', data=shapes_view, compression="lzf")
    
    # Save IoU sequence
    with h5py.File(data_dirs['iou_seq'] / f"{filename}.h5", 'w') as h5f:
        h5f.create_dataset('data', data=iou_seq, compression="lzf")
    
    # Save sketch images (all steps in one file)
    sketch_dir = data_dirs['sketch_img']
    sketch_dir.mkdir(exist_ok=True, parents=True)
    
    with h5py.File(sketch_dir / f"{filename}.h5", 'w') as h5f:
        sketch_group = h5f.create_group('steps')
        for i, step in enumerate(sketch_img):
            sketch_group.create_dataset(str(i), data=step, compression="lzf")

# Add subprocess timeout wrapper for get_iou_seq
def get_iou_seq_with_timeout(cfg, target_shape, shapes, timeout=100):
    """
    Run get_iou_seq in a subprocess with timeout to prevent hanging
    
    Args:
        cfg: Config object
        target_shape: Target shape
        shapes: List of shapes
        timeout: Timeout in seconds
        
    Returns:
        IoU sequence or None if timeout occurs
    """
    # Use a queue to get the result from the subprocess
    result_queue = multiprocessing.Queue()
    
    # Define worker function that puts result in queue
    def worker():
        try:
            result = get_iou_seq(cfg, target_shape, shapes)
            result_queue.put(result)
        except Exception as e:
            # Put the exception in the queue to propagate it
            result_queue.put(("ERROR", str(e), traceback.format_exc()))
    
    # Create and start the process
    process = multiprocessing.Process(target=worker)
    process.daemon = True  # Ensure process terminates when parent does    # But not allowed
    process.start()
    
    # Wait for process to complete or timeout
    start_time = time.time()
    while process.is_alive() and time.time() - start_time < timeout:
        time.sleep(0.1)  # Small sleep to prevent busy waiting
    
    # Check if process completed
    if process.is_alive():
        print(f"get_iou_seq timed out after {timeout} seconds")
        # Force terminate the process
        process.terminate()
        process.join(1)  # Give it a second to terminate
        if process.is_alive():
            # If it's still alive, use SIGKILL
            try:
                os.kill(process.pid, signal.SIGKILL)
            except:
                pass
        # Return default IoU sequence
        return np.zeros(len(shapes))
    
    # If process completed, get the result
    if not result_queue.empty():
        result = result_queue.get()
        # Check if an exception occurred
        if isinstance(result, tuple) and result[0] == "ERROR":
            print(f"Error in get_iou_seq: {result[1]}")
            print(result[2])  # Print traceback
            return np.zeros(len(shapes))
        return result
    
    # If queue is empty but process exited, something went wrong
    print("Process exited but no result was returned")
    return np.zeros(len(shapes))

def data_offload_new(cfg, filename, data_dirs):
    # # Skip if files already exist
    # if os.path.exists(data_dirs['gt_view'] / f"{filename}.h5"):
    #     return None, None
    
    try:
        # Get three views of ground truth
        gt_view = get_gt_view(cfg, filename, cfg.img_height, cfg.img_width)
        
        # Get vector data with normalization
        vec_raw, vec, cad_seq, vec_no_start, shape = get_vec_shape(cfg, filename)
        
        cutted = bool_op(cfg, vec_no_start, is_numerical=True)

        # Get three views
        shapes_view = get_three_view_shapes(cutted, cfg.img_height, cfg.img_width)

        cad_seq_no_start = CADSequence.from_vector(vec_no_start, is_numerical=False)
        cad_seq_no_start.denumericalize()
        # Get sketch images
        sketch_img = get_step_sketch_images(cfg, cad_seq_no_start, cfg.img_height, cfg.img_width, line_width=cfg.sketch_line_thickness)
        
        # Calculate mean and std
        mean, std = calculate_mean_std(gt_view, shapes_view, sketch_img)
        
        # Calculate IoU sequence with timeout protection
        if cfg.run_inference_reverse:
            # iou_seq = get_iou_seq_with_timeout(cfg, cutted[0], cutted[1:])
            iou_seq = get_iou_seq(cfg, cutted[0], cutted[1:])
        else:
            # iou_seq = get_iou_seq_with_timeout(cfg, shape, cutted)
            iou_seq = get_iou_seq(cfg, shape, cutted)
        
        # Save data to separate folders
        save_data(data_dirs, filename, gt_view, vec_raw, shapes_view, sketch_img, iou_seq)
        
        # # Also save mean and std to a separate file for process communication
        # result_file = Path(cfg.data_offload_root) / f"{filename}_result.json"
        # with open(result_file, 'w') as f:
        #     json.dump({
        #         "mean": mean.tolist() if isinstance(mean, np.ndarray) else mean,
        #         "std": std.tolist() if isinstance(std, np.ndarray) else std
        #     }, f)
        
        return mean, std
    
    except Exception as e:
        print(f"Error in {filename}, {e}")
        # Write error to file
        with open(Path(cfg.data_offload_root) / "errors.txt", "a") as f: 
            f.write(filename)
            f.write(str(e)+"\n")
        return None, None

# Process data in isolated processes - must be completely isolated
def process_single_file(file, config, data_dirs):
    try:
        # Process the file
        data_offload_new(config, file, data_dirs)
        # No need to return anything - results are saved to file
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        # Log error to a separate file
        error_file = Path(config.data_offload_root) / f"{file}_error.txt"
        with open(error_file, 'w') as f:
            f.write(str(e))
            f.write("\n")
            f.write(traceback.format_exc())

if __name__ == '__main__':
    # Use configuration class
    config = Config()
    
    # Create data directories
    data_dirs = create_data_dirs(config.data_offload_path)
    
    # Get file list
    with open(config.dataset_file, 'r') as f:
        dataset_file = json.load(f)
    filelist = dataset_file['train'] + dataset_file['validation'] + dataset_file['test']
    # Set random seed for reproducible file selection
    random.seed(42)
    # random choose 10% files
    filelist = random.sample(filelist, int(len(filelist)*0.1))
    
    # Process files in chunks to avoid memory issues
    batch_size = 256
    batches = [filelist[i:i + batch_size] for i in range(0, len(filelist), batch_size)]
    results = []
    
    # Maximum number of concurrent processes
    max_concurrent = 64
    
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)}")
        batch_results = []
        
        # Process files in smaller groups to control concurrency
        for i in range(0, len(batch), max_concurrent):
            current_files = batch[i:i+max_concurrent]
            processes = {}
            
            # Start processes for this group
            for file in current_files:
                process = multiprocessing.Process(target=process_single_file, args=(file, config, data_dirs))
                process.start()
                processes[file] = process
            
            # Wait for all processes to complete or timeout
            start_time = time.time()
            remaining_processes = list(processes.items())
            
            while remaining_processes and time.time() - start_time < 300:
                for file, process in list(remaining_processes):
                    if not process.is_alive():
                        remaining_processes.remove((file, process))
                time.sleep(0.1)
            
            # Handle results and cleanup any remaining processes
            for file, process in list(processes.items()):
                if process.is_alive():
                    print(f"Timeout processing {file}, terminating")
                    process.terminate()
                    process.join(1)
                    if process.is_alive():
                        try:
                            os.kill(process.pid, signal.SIGKILL)
                        except:
                            pass
                
                # # Collect results
                # try:
                #     result_file = Path(config.data_offload_root) / f"{file}_result.json"
                #     if result_file.exists():
                #         with open(result_file, 'r') as f:
                #             result = json.load(f)
                #         mean_val = np.array(result.get('mean')) if result.get('mean') else None
                #         std_val = np.array(result.get('std')) if result.get('std') else None
                #         batch_results.append((mean_val, std_val))
                #         os.remove(result_file)
                #     else:
                #         # File processing failed
                #         batch_results.append((None, None))
                # except Exception as e:
                #     print(f"Error retrieving results for {file}: {e}")
                #     batch_results.append((None, None))
        
        results.append(batch_results)
    
    # # test on one batch
    # batch = batches[0]
    # for file in batch:
    #     data_offload_new(config, file, data_dirs)


    # Save results to npz
    np.savez(Path(config.data_offload_root) / "results.npz", results=results)
    
    # Process and save mean/std statistics
    filtered_results = [result for result in results if result != (None, None)]
    filtered_results = filtered_results[:-1]  # Remove the last empty result
    
    results = np.array(filtered_results, dtype=float)
    mean_all = results[:, 0]
    std_all = results[:, 1]
    
    # Compute mean and std of all data
    mean_all = np.array([row for row in mean_all if row is not None])
    std_all = np.array([row for row in std_all if row is not None])
    
    std_all = (np.sum(std_all**2 + mean_all**2, axis=0)) / len(mean_all)
    mean_all = np.mean(mean_all, axis=0)
    std_all = np.sqrt(std_all - mean_all**2)
    
    # Save mean and std to json
    print(mean_all)
    print(std_all)
    with open(Path(config.data_offload_root) / "mean_std.json", "w") as f:
        json.dump({"mean": mean_all.tolist(), "std": std_all.tolist()}, f) 
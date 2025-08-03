import numpy as np
import cv2
from PIL import Image
import hashlib
import json
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
import torch

# Try to import denormalize_images function
try:
    from utils.img_utils import denormalize_images
except ImportError:
    # Fallback denormalization if utils not available
    def denormalize_images(img):
        return np.clip(img, 0, 1)


def calculate_image_hash(image: np.ndarray, method: str = 'phash', hash_size: int = 8) -> str:
    """
    Calculate hash value for a single image.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        method: Hash method ('phash', 'dhash', 'ahash')
        hash_size: Size of the hash (typically 8)
    
    Returns:
        Hash string
    """

    # image = image[0]
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            # Convert from RGB to grayscale if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                image = (image * 255).astype(np.uint8)
        elif len(image.shape) == 2:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    if method == 'phash':
        return calculate_phash(pil_image, hash_size)
    elif method == 'dhash':
        return calculate_dhash(pil_image, hash_size)
    elif method == 'ahash':
        return calculate_ahash(pil_image, hash_size)
    else:
        raise ValueError(f"Unknown hash method: {method}")


def calculate_phash(image: Image.Image, hash_size: int = 8) -> str:
    """Calculate perceptual hash (pHash)"""
    # Resize image
    image = image.resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Apply DCT
    pixels = np.asarray(image, dtype=np.float32)
    dct = cv2.dct(pixels)
    
    # Extract top-left 8x8
    dct_low = dct[:hash_size, :hash_size]
    
    # Calculate median
    median = np.median(dct_low)
    
    # Create hash
    hash_bits = dct_low > median
    return ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])


def calculate_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Calculate difference hash (dHash)"""
    # Resize to hash_size+1 x hash_size
    image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    image = image.convert('L')
    
    pixels = np.asarray(image)
    hash_bits = []
    
    # Compare adjacent pixels
    for row in range(hash_size):
        for col in range(hash_size):
            hash_bits.append(pixels[row, col] < pixels[row, col + 1])
    
    return ''.join(['1' if bit else '0' for bit in hash_bits])


def calculate_ahash(image: Image.Image, hash_size: int = 8) -> str:
    """Calculate average hash (aHash)"""
    # Resize and convert to grayscale
    image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    image = image.convert('L')
    
    pixels = np.asarray(image)
    avg = pixels.mean()
    
    # Create hash
    hash_bits = pixels > avg
    return ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hash strings"""
    if len(hash1) != len(hash2):
        raise ValueError("Hash strings must have the same length")
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def calculate_hash_similarity(hash1: str, hash2: str) -> float:
    """
    Calculate similarity between two hashes (0-1, where 1 is identical)
    """
    distance = hamming_distance(hash1, hash2)
    max_distance = len(hash1)
    similarity = 1.0 - (distance / max_distance)
    return similarity


def calculate_three_view_hash_similarity(
    current_views: List[np.ndarray], 
    target_views: List[np.ndarray],
    method: str = 'phash',
    hash_size: int = 8
) -> float:
    """
    Calculate average hash similarity between two sets of three views.
    
    Args:
        current_views: List of 3 current view images
        target_views: List of 3 target view images  
        method: Hash method
        hash_size: Hash size
    
    Returns:
        Average similarity across all three views (0-1)
    """
    if len(current_views) != 3 or len(target_views) != 3:
        raise ValueError("Both view sets must contain exactly 3 images")
    
    similarities = []
    
    for i in range(3):
        # Calculate hashes
        current_hash = calculate_image_hash(current_views[i], method, hash_size)
        target_hash = calculate_image_hash(target_views[i], method, hash_size)
        
        # Calculate similarity
        similarity = calculate_hash_similarity(current_hash, target_hash)
        similarities.append(similarity)
    
    # Return average similarity
    return np.mean(similarities)


class HashSimilarityCalculator:
    """
    Class for managing hash similarity calculations with baseline normalization.
    """
    
    def __init__(self, method: str = 'phash', hash_size: int = 8):
        self.method = method
        self.hash_size = hash_size
        
        # Store baseline values per batch item to handle batch_size > 1
        self.baselines = {}  # batch_idx -> {'blank_to_target': float, 'target_to_target': 1.0}
    
    def calculate_baselines(self, blank_views: List[np.ndarray], target_views: List[np.ndarray], batch_idx: int = 0):
        """
        Calculate and store baseline similarity values for normalization.
        
        Args:
            blank_views: List of 3 blank view images (from data pipeline)
            target_views: List of 3 target view images
            batch_idx: Batch index to identify this specific shape
        """
        # Calculate blank to target similarity using actual blank views from data pipeline
        blank_to_target_similarity = calculate_three_view_hash_similarity(
            blank_views, target_views, self.method, self.hash_size
        )
        
        # Store baselines for this batch item
        self.baselines[batch_idx] = {
            'blank_to_target': blank_to_target_similarity,
            'target_to_target': 1.0  # Perfect match
        }
    
    def calculate_normalized_similarity(self, current_views: List[np.ndarray], 
                                      target_views: List[np.ndarray], batch_idx: int = 0) -> float:
        """
        Calculate normalized similarity between current and target views.
        
        The similarity is normalized so that:
        - Blank views vs target views → close to 0
        - Perfect match → close to 1
        
        Args:
            current_views: List of 3 current view images
            target_views: List of 3 target view images
            batch_idx: Batch index to identify which baseline to use
        
        Returns:
            Normalized similarity (0-1)
        """
        if batch_idx not in self.baselines:
            raise ValueError(f"Baseline values not calculated for batch_idx {batch_idx}. Call calculate_baselines first.")
        
        # Calculate raw similarity
        raw_similarity = calculate_three_view_hash_similarity(
            current_views, target_views, self.method, self.hash_size
        )
        
        # Get baselines for this batch item
        blank_to_target_similarity = self.baselines[batch_idx]['blank_to_target']
        target_to_target_similarity = self.baselines[batch_idx]['target_to_target']
        
        # Normalize: (raw - blank_baseline) / (target_baseline - blank_baseline)
        # This maps blank_baseline → 0 and target_baseline → 1
        if target_to_target_similarity - blank_to_target_similarity == 0:
            # Edge case: if blank and target similarities are the same
            return 1.0 if raw_similarity >= target_to_target_similarity else 0.0
        
        normalized = (raw_similarity - blank_to_target_similarity) / \
                    (target_to_target_similarity - blank_to_target_similarity)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, normalized))
    
    def clear_baselines(self):
        """Clear all stored baselines (useful for memory management)"""
        self.baselines.clear()
    
    @property
    def blank_to_target_similarity(self):
        """For backward compatibility - returns baseline for batch_idx=0"""
        return self.baselines.get(0, {}).get('blank_to_target', None)
    
    @property 
    def target_to_target_similarity(self):
        """For backward compatibility - returns baseline for batch_idx=0"""
        return self.baselines.get(0, {}).get('target_to_target', 1.0)


def convert_tensor_views_to_numpy(views: Union[torch.Tensor, List[torch.Tensor]], 
                                 denormalize: bool = False) -> List[np.ndarray]:
    """
    Convert tensor views to numpy arrays for hash calculation.
    
    Args:
        views: Either a tensor of shape (3, H, W, C) or list of 3 tensors
        denormalize: Whether to denormalize images (if they were normalized during training)
    
    Returns:
        List of 3 numpy arrays
    """
    if isinstance(views, torch.Tensor):
        if views.dim() == 4 and views.shape[0] == 3:
            # Shape (3, H, W, C)
            view_list = [views[i] for i in range(3)]
        else:
            raise ValueError(f"Expected tensor shape (3, H, W, C), got {views.shape}")
    else:
        view_list = views
    
    numpy_views = []
    for view in view_list:
        if isinstance(view, torch.Tensor):
            view_np = view.detach().cpu().numpy()
        else:
            view_np = view
        
        # Denormalize if needed (assuming normalization was done with mean/std)
        if denormalize:
            # Use the denormalize_images function if available
            view_np = denormalize_images(view_np)[0]
        else:
            # Ensure values are in [0,1] range
            view_np = np.clip(view_np, 0, 1)
        
        numpy_views.append(view_np)
    
    return numpy_views 
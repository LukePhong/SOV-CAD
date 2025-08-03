import numpy as np

def normalize_images(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Normalize images with given mean and standard deviation.
    
    Args:
        images (numpy.ndarray): Input images of shape (N, H, W, 3).
        mean (list or numpy.ndarray): Mean for each channel (R, G, B).
        std (list or numpy.ndarray): Standard deviation for each channel (R, G, B).
    
    Returns:
        numpy.ndarray: Normalized images.
    """
    mean = np.array(mean).reshape(1, 1, 1, 3)
    std = np.array(std).reshape(1, 1, 1, 3)
    return (images - mean) / std

def denormalize_images(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Denormalize images with given mean and standard deviation.
    
    Args:
        images (numpy.ndarray): Normalized images of shape (N, H, W, 3).
        mean (list or numpy.ndarray): Mean for each channel (R, G, B).
        std (list or numpy.ndarray): Standard deviation for each channel (R, G, B).
    
    Returns:
        numpy.ndarray: Denormalized images.
    """
    mean = np.array(mean).reshape(1, 1, 1, 3)
    std = np.array(std).reshape(1, 1, 1, 3)
    return (images * std) + mean
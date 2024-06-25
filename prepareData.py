import numpy as np
from scipy.linalg import lstsq

def prepareData(images, ambient):
    normalized_images = np.maximum(images - ambient, 0)
    
    max_intensity = np.max(normalized_images)
    if max_intensity > 0:
        normalized_images /= max_intensity
    
    return normalized_images
import numpy as np
from scipy.linalg import lstsq

def photometricStereo(normalized_images, light_dirs, num_paths=1):
    h, w, num_images = normalized_images.shape
    albedo = np.zeros((h, w))
    normals = np.zeros((h, w, 3))

    for _ in range(num_paths):
        random_indices = np.random.choice(num_images, num_images, replace=False)
        for i in range(h):
            for j in range(w):
                pixel_values = normalized_images[i, j, random_indices]
                G = np.transpose(light_dirs[random_indices])
                albedo[i, j], normals[i, j, _], _ = lstsq(G, pixel_values)
                normals[i, j] /= np.linalg.norm(normals[i, j])

    albedo /= num_paths
    normals /= num_paths
    
    return albedo, normals
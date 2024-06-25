import numpy as np
from scipy.linalg import lstsq
import os
import matplotlib.pyplot as plt

def getpgmraw(filename):
    with open(filename, 'rb') as f:
        magic_number = f.readline().decode('utf-8').strip()
        if magic_number != 'P5':
            raise ValueError("Invalid PGM file format")

        width, height = map(int, f.readline().decode('utf-8').split())
        max_pixel_value = int(f.readline().decode('utf-8'))

        if max_pixel_value > 255:
            raise ValueError("Invalid maximum pixel value in PGM file")

        image_data = np.frombuffer(f.read(), dtype=np.uint8)

        image_data = image_data.reshape((height, width))

    return image_data


def prepareData(images, ambient):
    normalized_images = np.maximum(images - ambient, 0)
    
    max_intensity = np.max(normalized_images)
    if max_intensity > 0:
        normalized_images /= max_intensity
    
    return normalized_images

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

def getSurface(normals):
    h, w, _ = normals.shape
    height_map_row_column = np.zeros((h, w))
    height_map_column_row = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                height_map_row_column[i, j] = 0
                height_map_column_row[i, j] = 0
            elif i == 0:
                height_map_row_column[i, j] = height_map_row_column[i, j-1] + normals[i, j-1, 2]
                height_map_column_row[i, j] = height_map_column_row[i, j-1] + normals[i, j-1, 2]
            elif j == 0:
                height_map_row_column[i, j] = height_map_row_column[i-1, j] + normals[i-1, j, 2]
                height_map_column_row[i, j] = height_map_column_row[i-1, j] + normals[i-1, j, 2]
            else:
                height_map_row_column[i, j] = (height_map_row_column[i-1, j] + normals[i-1, j, 2] +
                                                height_map_row_column[i, j-1] + normals[i, j-1, 2]) / 2
                height_map_column_row[i, j] = (height_map_column_row[i-1, j] + normals[i-1, j, 2] +
                                                height_map_column_row[i, j-1] + normals[i, j-1, 2]) / 2
    
    return height_map_row_column, height_map_column_row

def shapeFromShading(images, light_dirs, ambient):
    normalized_images = prepareData(images, ambient)
    
    albedo, normals = photometricStereo(normalized_images, light_dirs)
    
    height_map_row_column, height_map_column_row = getSurface(normals)
    
    return albedo, normals, height_map_row_column, height_map_column_row

def loadFaceImages(data_folder):
    subjects = os.listdir(data_folder)
    num_subjects = len(subjects)
    images = np.zeros((192, 168, 64, num_subjects), dtype=np.uint8)
    ambient_images = np.zeros((192, 168, num_subjects), dtype=np.uint8)
    light_dirs = []

    for subject_idx, subject in enumerate(subjects):
        subject_folder = os.path.join(data_folder, subject)
        light_dir_file = os.path.join(subject_folder, 'light_directions.txt')
        light_dirs.append(np.loadtxt(light_dir_file))

        for light_idx in range(64):
            image_file = os.path.join(subject_folder, f'img_{light_idx + 1}.pgm')
            images[:, :, light_idx, subject_idx] = getpgmraw(image_file)

        ambient_file = os.path.join(subject_folder, 'ambient.pgm')
        ambient_images[:, :, subject_idx] = getpgmraw(ambient_file)

    return images, ambient_images, light_dirs

# Example usage:
data_folder = ''
images, ambient_images, light_dirs = loadFaceImages(data_folder)

# Visualize the first image in grayscale
plt.imshow(images[:, :, 0, 0], cmap='gray')
plt.show()


# Assuming you have images, light directions, and ambient lighting defined
# albedo, normals, height_map_row_column, height_map_column_row = shapeFromShading(images, light_dirs, ambient)

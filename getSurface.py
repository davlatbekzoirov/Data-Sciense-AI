import numpy as np

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


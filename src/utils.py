import numpy as  np 
#from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_distances
from tslearn.metrics import dtw

def cosine_dist(x, y):
    return cosine_distances([x],[y])[0,0]

def eu(x, y):
    return np.sqrt(np.sum((x-y)**2))

def abs_dist(x, y):
    return sum(np.abs(x-y))

def dtw(x, y):
    return dtw(y, x)


def detect_knee_point(values, indices):
    """
    From:
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    # get coordinates of all the points
    #print(values)
    #print(indices)
    n_points = len(values)
    #print(n_points)
    all_coords = np.vstack((range(n_points), values)).T
    # get the first point
    first_point = all_coords[0]
    # get vector between first and last point - this is the line
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # distance to line is the norm of vec_to_line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # knee/elbow is the point with max distance value
    knee_idx = np.argmax(dist_to_line)
    knee  = values[knee_idx] 
    #print(f"Knee Value: {values[knee_idx]}, {knee_idx}") 
    
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem>knee]
    best_dis = [elem for (elem, idx) in zip(values, indices) if elem>knee]
    if len(best_dims)==0:
        return [knee_idx], knee_idx

    return best_dims, best_dis, knee_idx

def detect_knee_point2(values, indices, second = False):
    """
    From:
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    # get coordinates of all the points
    #print(values)
    #print(indices)
    n_points = len(values)
    #print(n_points)
    all_coords = np.vstack((range(n_points), values)).T
    # get the first point
    first_point = all_coords[0]
    # get vector between first and last point - this is the line
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # distance to line is the norm of vec_to_line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # knee/elbow is the point with max distance value
    knee_idx = np.argmax(dist_to_line)
    knee  = values[knee_idx] 
    #print(f"Knee Value: {values[knee_idx]}, {knee_idx}") 
    
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem>knee]
    #print(best_dims)
    best_dis = [elem for (elem, idx) in zip(values, indices) if elem>knee]
    if len(best_dims)==0:
        return [knee_idx], knee_idx
    
    if second:
        #print(":", best_dims, best_dis, knee_idx)
        best_dims1, best_dis1, knee_idx1= detect_knee_point(values[knee_idx:], indices[knee_idx:])
        #print("$", best_dims.append(best_dims1, best_dis.extend(best_dis1), knee_idx))
        return best_dims + best_dims1, best_dis+best_dis1, knee_idx1

    return best_dims, best_dis, knee_idx

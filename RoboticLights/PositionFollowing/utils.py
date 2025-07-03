import numpy as np

"""
# for multiple vectors
def cart2spherical(vector):
    
    print("cart2spherical begin")
    print("vector s ", vector.shape)
    
    # normalize direction
    vector_norm = vector / np.linalg.norm(vector, axis=1, keepdims=True)
    
    #print("vector_norm s ", vector_norm.shape)
    
    # inclination
    elevation = np.arccos( vector_norm[:, 2])
    
    #print("elevation s ", elevation.shape)

    # azimuth
    azimuth = np.arctan2(vector_norm[:, 1],vector_norm[:, 0])
    
    #print("azimuth s ", azimuth.shape)

    return np.stack((azimuth, elevation), axis=1)
"""

# for single vector only
def cart2spherical(vector):
    
    #print("cart2spherical begin")
    #print("vector s ", vector.shape)
    
    # normalize direction
    vector_norm = vector / np.linalg.norm(vector)
    
    #print("vector_norm s ", vector_norm.shape)
    
    # inclination
    elevation = np.arccos( vector_norm[2])
    
    #print("elevation s ", elevation.shape)

    # azimuth
    azimuth = np.arctan2(vector_norm[1],vector_norm[0])
    
    #print("azimuth s ", azimuth.shape)

    return np.array((azimuth, elevation))
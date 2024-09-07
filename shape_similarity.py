import gpytoolbox as gpy
import numpy as np
import igl


def chamfer_distance(V1, F1, V2, F2, N=10000):
    """
    Calculates the chamfer distance between two 3D meshes.

    Parameters:
    - V1 (numpy.ndarray): Vertices of the first mesh.
    - F1 (numpy.ndarray): Faces of the first mesh.
    - V2 (numpy.ndarray): Vertices of the second mesh.
    - F2 (numpy.ndarray): Faces of the second mesh.
    - N (int): Number of points to sample on the meshes (default: 1000000).

    Returns:
    - float: The chamfer distance between the two meshes.
    """
    # Sample points on the two meshes
    P1 = gpy.random_points_on_mesh(V1, F1, N)
    P2 = gpy.random_points_on_mesh(V2, F2, N)
    # Compute the squared distances between the two sets of points
    sqrD1 = gpy.squared_distance(P1, P2, use_aabb=True, use_cpp=True)[0]
    sqrD2 = gpy.squared_distance(P2, P1, use_aabb=True, use_cpp=True)[0]
    return np.sqrt(np.mean(sqrD1)) + np.sqrt(np.mean(sqrD2))


def hausdorff_distance(V1, F1, V2, F2):
    """
    Calculates the Hausdorff distance between two 3D meshes.

    Parameters:
    - V1 (numpy.ndarray): Vertices of the first mesh.
    - F1 (numpy.ndarray): Faces of the first mesh.
    - V2 (numpy.ndarray): Vertices of the second mesh.
    - F2 (numpy.ndarray): Faces of the second mesh.

    Returns:
    - float: The Hausdorff distance between the two meshes.
    """
    D = igl.hausdorff(V1, F1, V2, F2)
    return D

def normalize(V):
    """
    Normalizes the vertices of a 3D mesh.
    
    Parameters:
    - V (numpy.ndarray): Vertices of the mesh.
    
    Returns:
    - numpy.ndarray: The normalized vertices.
    """
    # Center the vertices by subtracting the mean
    center = np.mean(V, axis=0)
    V_centered = V - center
    
    # Compute the scale factor to fit the vertices in a unit sphere
    max_distance = np.max(np.linalg.norm(V_centered, axis=1))
    
    # Normalize the vertices
    V_normalized = V_centered / max_distance
    
    return V_normalized

def rigid_alignment(V1, F1, V2, F2, n_samples=10000, max_iters=1000, apply_transform=True):
    """
    Aligns two 3D meshes using the iterative closest point (ICP) algorithm.
    
    Parameters:
    - V1 (numpy.ndarray): Vertices of the first mesh.
    - F1 (numpy.ndarray): Faces of the first mesh.
    - V2 (numpy.ndarray): Vertices of the second mesh.
    - F2 (numpy.ndarray): Faces of the second mesh.
    - n_samples (int): Number of points to sample on the meshes (default: 10000).
    - max_iters (int): Maximum number of iterations (default: 100).
    - apply_transform (bool): Whether to apply the transformation to the second mesh (default: True).
    
    Returns:
    - numpy.ndarray: The aligned vertices of the second mesh.
    """
    R, t = igl.iterative_closest_point(V1, F1, V2, F2, num_samples=n_samples, max_iters=max_iters)
    # R is a 3x3 rotation matrix and t is a 3D translation vector
    if apply_transform:
        return V2 @ R.T + t
    return R, t


def shape_matching(V1, F1, V2, F2):
    """
    Matches two 3D meshes using the shape matching algorithm.
    
    Parameters:
    - V1 (numpy.ndarray): Vertices of the first mesh.
    - F1 (numpy.ndarray): Faces of the first mesh.
    - V2 (numpy.ndarray): Vertices of the second mesh.
    - F2 (numpy.ndarray): Faces of the second mesh.
    
    Returns:
    - numpy.ndarray: Vertices of the first mesh.
    - numpy.ndarray: Faces of the first mesh.
    - numpy.ndarray: Vertices of the second mesh.
    - numpy.ndarray: Faces of the second mesh.
    """
    # Normalize the vertices of the two meshes
    V1 = normalize(V1)
    V2 = normalize(V2)
    # Match the two meshes
    V2 = rigid_alignment(V1, F1, V2, F2)
    return V1, F1, V2, F2

def load_mesh(path):
    """
    Loads a 3D mesh from a file.
    
    Parameters:
    - path (str): Path to the mesh file.
    
    Returns:
    - numpy.ndarray: Vertices of the mesh.
    - numpy.ndarray: Faces of the mesh.
    """
    V, F = igl.read_triangle_mesh(path)
    # print(V.shape[0], np.max(F))
    # V, F = igl.remove_duplicates(V, F, 1e-7)
    # V, F, _, _ = igl.remove_unreferenced(V, F)
    return V, F
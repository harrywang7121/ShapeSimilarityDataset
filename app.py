from shape_similarity_new import *
import numpy as np

V1, F1 = load_mesh("data/mesh.obj")
V2, F2 = load_mesh("data/orig.obj")

V1, F1, V2, F2 = shape_matching(V1, F1, V2, F2)

chamferD = chamfer_distance(V1, F1, V2, F2)
hausdorffD = hausdorff_distance(V1, F1, V2, F2)

print(f"Chamfer Distance: {chamferD:.4f}")
print(f"Hausdorff Distance: {hausdorffD:.4f}")
import os
from shape_similarity_new import *
import numpy as np
import pandas as pd
import re


def batch_compare_shapes(base_path):
    while True:  # Infinite loop
        directories = [d for d in os.listdir(base_path) if
                       re.match(r'^\d{3}$', d) and os.path.isdir(os.path.join(base_path, d))]
        for directory in directories:
            base_folder = os.path.join(base_path, directory)
            base_files = [f for f in os.listdir(base_folder) if f.endswith('.obj')]
            if len(base_files) != 1:
                print(f"Skipping {directory} as it does not have exactly one base obj file.")
                continue
            base_file = os.path.join(base_folder, base_files[0])
            V_base, F_base = load_mesh(base_file)

            subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if
                          os.path.isdir(os.path.join(base_folder, d))]
            for subfolder in subfolders:
                mesh_file = os.path.join(subfolder, '0', 'mesh.obj')
                if os.path.exists(mesh_file):
                    V_mesh, F_mesh = load_mesh(mesh_file)
                    V_base, F_base, V_mesh, F_mesh = shape_matching(V_base, F_base, V_mesh, F_mesh)
                    chamferD = chamfer_distance(V_base, F_base, V_mesh, F_mesh)
                    hausdorffD = hausdorff_distance(V_base, F_base, V_mesh, F_mesh)

                    results_df = pd.DataFrame({
                        "Base Folder": [directory],
                        "Mesh Folder": [subfolder],
                        "Chamfer Distance": [chamferD],
                        "Hausdorff Distance": [hausdorffD]
                    })

                    # Construct the filename with incrementing numbers to avoid overwrites
                    subfolder_path = os.path.join(subfolder, '0')
                    file_index = 1
                    results_path = os.path.join(subfolder_path, f'results_{file_index}.csv')
                    while os.path.exists(results_path):
                        file_index += 1
                        results_path = os.path.join(subfolder_path, f'results_{file_index}.csv')

                    # Save the results
                    results_df.to_csv(results_path, index=False)
                    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    dataset_path = '.'
    batch_compare_shapes(dataset_path)

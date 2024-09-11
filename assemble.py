import os
import pandas as pd
from glob import glob

top_directory = '.'

result_frames = []

for i in range(100):
    base_folder = f"{i:03d}"
    base_path = os.path.join(top_directory, base_folder)

    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir, '0')

        csv_files = glob(os.path.join(subdir_path, 'results_*.csv'))

        min_chamfer = float('inf')
        min_hausdorff = float('inf')

        for file in csv_files:
            df = pd.read_csv(file)
            current_min_chamfer = df['Chamfer Distance'].min()
            current_min_hausdorff = df['Hausdorff Distance'].min()

            if current_min_chamfer < min_chamfer:
                min_chamfer = current_min_chamfer
            if current_min_hausdorff < min_hausdorff:
                min_hausdorff = current_min_hausdorff

        result_frames.append(pd.DataFrame({
            'Base Folder': [base_folder],
            'Mesh Folder': [f"./{base_folder}/{subdir}"],
            'Chamfer Distance': [min_chamfer],
            'Hausdorff Distance': [min_hausdorff]
        }))
        print("finished processing:" + subdir_path)

final_results = pd.concat(result_frames, ignore_index=True)

final_results.to_csv('./final_results.csv', index=False)

import os
from typing import List, Tuple


def load_dataset(root_folder: str) -> Tuple[List[str], int]:
    folders = os.listdir(root_folder)
    dataset = []
    labels = []

    for folder in folders:
        folder_path = os.path.join(root_folder, folder)
        files = os.listdir(folder_path)
        files = [os.path.join(folder_path, file) for file in files]
        dataset += files

        for file in files:
            filename = os.path.splitext(file)[0]
            if filename.endswith("cafe"):
                labels.append(1)
            elif filename.endswith("highway"):
                labels.append(2)
            elif filename.endswith("park"):
                labels.append(3)
            else:
                labels.append(0)

    return dataset, labels
    
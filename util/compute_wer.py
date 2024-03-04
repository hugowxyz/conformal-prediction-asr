import os
import jiwer
import pandas as pd
from load_dataset import load_dataset

if __name__ == "__main__":
    wer_root = "2023-12-04-HMI-dataset-predictions"

    data = pd.read_excel(os.path.join("..", "HMI_questions_and_intents.xlsx"))
    ground_truth_labels = data['sentence']

    dataset, labels = load_dataset(os.path.join("..", "2023-12-04-HMI-dataset-predictions"))
    dataset.sort()

    for filepath in dataset:
        folder = filepath.split("/")[2]

        filename = os.path.basename(filepath)
        r = os.path.splitext(filename)[0].split("-")

        if len(r) == 3:
            audio, modifier, preds = r
            wer_filepath = os.path.join("..", wer_root, folder, f"{audio}-{modifier}-wer.txt")
        else:
            audio, preds = r
            wer_filepath = os.path.join("..", wer_root, folder, f"{audio}-wer.txt")
            
        if preds != "preds":
            continue
        
        with open(filepath, "r") as preds_file:
            with open(wer_filepath, "w") as wer_file:
                predictions = preds_file.readlines()
                true_label_idx = int(audio[-2:])
                true_label = ground_truth_labels[true_label_idx]

                for idx, prediction in enumerate(predictions):
                    wer = jiwer.wer(true_label, prediction)
                    wer_file.write(str(wer) + "\n")

        print(f"Wrote to {wer_filepath}")
            

        
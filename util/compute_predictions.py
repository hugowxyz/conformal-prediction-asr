import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import librosa
import torch
from datetime import datetime

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return exp_x / np.sum(exp_x)

def compute_predictions(audio_file_path, predictions_out_path, scores_out_path, num_return_sentences=50):
    audio_sample, sampling_rate = librosa.load(audio_file_path, sr=None)

    target_sampling_rate=16000
    waveform = librosa.resample(audio_sample, orig_sr=sampling_rate, target_sr=target_sampling_rate)

    input_features = processor(
    waveform, sampling_rate=target_sampling_rate, return_tensors="pt"
    ).input_features

    # input_features = input_features.to("cuda")
    predicted_ids = model.generate(input_features, num_beams=num_return_sentences, output_scores=True, num_return_sequences=num_return_sentences, return_dict_in_generate=True, max_time=10)
    transcriptions = processor.batch_decode(predicted_ids['sequences'], skip_special_tokens=True)
    scores = softmax(np.array(predicted_ids.sequences_scores.to("cpu")))

    with open(predictions_out_path, "w") as file:
        for line in transcriptions:
            file.write(line + "\n")

    with open(scores_out_path, "w") as file:
        for line in scores:
            file.write(str(line) + "\n")

    print(f"[{datetime.now().isoformat()}] Wrote {predictions_out_path} and {scores_out_path} for {audio_file_path}")


if __name__ == "__main__":
    root1 = "2023-12-04-HMI-dataset-predictions"
    root = "2023-12-04-HMI-dataset"
    rp = os.path.join("..", root)
    folders = sorted(os.listdir(rp))
    for folder in folders:
        fp = os.path.join(rp, folder)
        files = sorted(os.listdir(fp))
        for file in files:
            filename = os.path.splitext(file)[0]
            
            path = os.path.join("..", root, folder, file)
            folder_path = os.path.join("..", root1, folder)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            preds_path = os.path.join(folder_path ,filename + "-preds.txt")
            scores_path = os.path.join(folder_path, filename + "-scores.txt")
            
            print(path, preds_path, scores_path)

            compute_predictions(path, preds_path, scores_path)
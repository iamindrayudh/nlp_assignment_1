import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- CONFIGURATION ---
MODEL_PATH = "./wav2vec2-large-100k-voxpopuli-ft-Common-Voice_plus_TTS-Dataset-russian"
DATASET_PATH = "./cv-corpus-7.0-2021-07-21/ru"
TSV_FILE = "test.tsv"
TARGET_SAMPLE_RATE = 16000
NUM_FILES_TO_TEST = 1000 # <-- New variable to easily change the number of files
OUTPUT_FILE = "evaluation_results_CV+TTS_ru.txt" # <-- ADDED: Name for the output file
# ---------------------

def main():
    # --- ADDED: Auto-detect CUDA or CPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # --------------------------------------

    # 1. Load Model and Processor
    print("Loading model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    
    # --- MODIFIED: Move model to the detected device ---
    model.to(device)
    # -------------------------------------------------
    
    print("Model and processor loaded.")

    # 2. Load the TSV manifest file
    tsv_path = os.path.join(DATASET_PATH, TSV_FILE)
    print(f"Loading dataset manifest from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    df = df.head(NUM_FILES_TO_TEST) 
    
    references = []
    predictions = []

    # --- ADDED: Write configuration to the output file ---
    with open(OUTPUT_FILE, 'w') as f:
        f.write("--- EVALUATION CONFIGURATION ---\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Dataset Path: {DATASET_PATH}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of files tested: {NUM_FILES_TO_TEST}\n")
        f.write("--------------------------------\n\n")
    # ---------------------------------------------------

    # 3. Loop through dataset and evaluate
    print(f"Starting evaluation on {NUM_FILES_TO_TEST} files...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            reference_text = str(row['sentence']).lower()
            audio_file = row['path']
            
            # Construct the full path to the audio clip
            audio_path = os.path.join(DATASET_PATH, "clips", audio_file)

            # Load and preprocess the audio
            speech_array, sampling_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sampling_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=TARGET_SAMPLE_RATE)
                speech_array = resampler(speech_array)
            
            # Get model prediction
            input_values = processor(speech_array.squeeze().numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_values
            
            # --- ADDED: Move input data to the same device as the model ---
            input_values = input_values.to(device)
            # ------------------------------------------------------------
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentence = processor.batch_decode(predicted_ids)[0].lower()

            # Store results
            references.append(reference_text)
            predictions.append(predicted_sentence)

        except Exception as e:
            print(f"Skipping file {audio_file} due to error: {e}")

    # 4. Calculate final WER
    print("Evaluation complete. Calculating WER...")
    error = wer(references, predictions)

    print("\n--- FINAL RESULT ---")
    print(f"Word Error Rate (WER): {error:.4f}")
    print("--------------------")

    # --- ADDED: Append the final result to the output file ---
    with open(OUTPUT_FILE, 'a') as f:
        f.write("--- FINAL RESULT ---\n")
        f.write(f"Word Error Rate (WER): {error:.4f}\n")
        f.write("--------------------\n")
    
    print(f"Results have been saved to {OUTPUT_FILE}")
    # -------------------------------------------------------


if __name__ == "__main__":
    main()
# ASR Evaluation for Portuguese and Russian

This project evaluates a fine-tuned Wav2Vec 2.0 model on the Common Voice dataset for Portuguese and Russian.

## Setup

1. **Codebase**: Clone the Wav2Vec-Wrapper repo from (https://github.com/Edresson/Wav2Vec-Wrapper/tree/main)

2.  **Model Checkpoint**: Download the model from (https://github.com/Edresson/Wav2Vec-Wrapper/tree/main/Papers/TTS-Augmentation)

3.  **Dataset**: Download the Common Voice 7.0 Portuguese and Russian dataset from the [official Mozilla Voice website](https://voice.mozilla.org/en/datasets).

## Run Evaluation
```sh
python eval_wer.py
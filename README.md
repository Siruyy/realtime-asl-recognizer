# ASL Static Letter Recognizer (A–Y, no J/Z)

Static ASL fingerspelling recognizer using **MediaPipe + TensorFlow**.

- Included: **24 static letters** (`A` to `Y`, excluding `J` and `Z`)
- Excluded: `J`, `Z` (motion-based letters)

## Datasets

Use these published datasets instead of local repo copies:

- Kaggle: https://www.kaggle.com/datasets/siruyyy/asl-hand-landmarks-24-letters-v1-a-y-no-jz/data
- Hugging Face: https://huggingface.co/datasets/Siruyy/asl-static-landmarks-v1

## What is in this repo

- `src/` — training and inference scripts
- `models/` — model artifacts used by local predictor scripts
- `requirements.txt` — dependencies for local development

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

`pip install -r requirements.txt`

3. Run live local prediction:

`python3 src/predictor_static.py`

### Controls

- `q`: quit
- `+` / `-`: adjust confidence threshold

## Notes

- Current pipeline is optimized for **single-hand static poses**.
- For full alphabet support, add a temporal model for `J` and `Z`.


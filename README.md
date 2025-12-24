# Xenoglaux-AI

**Project**: Xenoglaux-AI is a personal research / prototype repository containing scripts and utilities for building, training, and running a small language/math model. The project is under active development.

**Status**: under development — work in progress. The codebase contains training and inference scripts, data processing utilities, and a lightweight app interface. Expect breaking changes and ongoing improvements.

**Data & Models**: All training data and the trained model referenced in this repository are created and produced from scratch by the author (no external datasets). Data and model files are not bundled in this repo; training artifacts are created locally by the developer.

**Quick repository overview**

- **Purpose**: Tools and scripts to preprocess custom data, train a toy/experimental language/math model, and run inference. This repo is intended for experimentation and development rather than production use.
- **Author / Owner**: Owlicorn (repository owner)
- **License**: See the included `LICENSE` file.

**Structure & important files**

- [app.py](app.py) : Minimal app/starter script for demoing the model or serving inference (prototype).
- [config.py](config.py) : Project configuration and hyperparameters (used by training/inference scripts).
- [data_loader.py](data_loader.py) : Utilities to read and batch input data.
- [data_validator.py](data_validator.py) : Sanity checks and validation routines for the custom dataset.
- [fix_math_data.py](fix_math_data.py) : Helpers to clean/normalize math-related textual data.
- [inference.py](inference.py) : Script to run model inference on inputs (prototype usage).
- [maths_train.py](maths_train.py) : Experimental training utilities focused on math-related datasets.
- [model.py](model.py) : Model definition and network architecture (small/experimental).
- [prompt.json](prompt.json) : Example prompts / templates used during inference or evaluation.
- [quick_text.py](quick_text.py) : Fast helpers for text I/O and quick tests.
- [tokenizer.py](tokenizer.py) : Tokenization utilities for preparing data.
- [trainer.py](trainer.py) : Main training script / orchestration. Intended to be adapted by the developer.
- [data/lang_model.txt](data/lang_model.txt) : Example or in-progress datasource (text). This repo currently expects the developer to supply or generate the working dataset.
- [templates/](templates/) : Simple HTML templates for demonstration UI ([templates/index.html](templates/index.html), [templates/chat.html](templates/chat.html), [templates/doc.html](templates/doc.html)).
- [requirements.txt](requirements.txt) : Python dependencies used by the project.

**Getting started (developer)**

1. **Easy Setup**: Use the main script which auto-installs dependencies:

```bash
python main.py train
```

Or for manual setup, create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your dataset. This project assumes you create and manage the training data locally (all data in this repo is authored by the developer).

3. **Training**: Run training with auto-dependency installation:

```bash
# Basic training
python main.py train

# Training with streaming mode for large datasets
python main.py train --streaming

# Training with custom parameters
python main.py train --epochs 10 --batch-size 32 --lr 0.001
```

4. **Inference/Chat**: Run interactive inference:

```bash
# Interactive chat mode
python main.py chat

# Inference with custom model/tokenizer paths
python main.py infer --model models/xenoglaux_model.pt --tokenizer models/tokenizer
```

**Device Support**: The model automatically detects and uses the best available hardware:
- **TPU** (Google Colab, Kaggle TPU) - fastest training
- **GPU** (CUDA) - fast training  
- **CPU** - fallback for any system

**Manual training** (legacy method):

```bash
python trainer.py
```

4. Run inference (prototype example):

```bash
python inference.py
```

Notes: The training and inference scripts are experimental and may require editing `config.py` or passing additional CLI arguments. Check the scripts for available options.

**Development notes & expectations**

- The repository is actively developed; APIs, file formats, and CLIs may change.
- The dataset and trained weights are private and produced by the repository owner; they are not included here.
- Tests, CI, and packaging are not yet provided — contributions should focus on stabilizing training pipelines and documenting reproducible runs.

**How you can help / contribute**

- If you collaborate with the author, open issues describing reproducible steps, or submit focused pull requests that include tests and documentation.

**Contact**

- For questions about the code or data provenance, contact the repository owner (Owlicorn) via the project GitHub.

---
_This README reflects the current state: experimental and under active development._

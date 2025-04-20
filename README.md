
# Audio Declip Project

This repository provides code and tools for training and evaluating models for **audio declipping**. It supports using pre-generated datasets or generating datasets from custom `.wav` files.

---

## 📁 Repository Structure

```
Audio_Declip/
├── data/
│   ├── train/                  # Training .wav files
│   ├── test/                   # Testing .wav files
│   └── pkl_data/               # Preprocessed training data (.pkl)
├── src/                        # Core Python modules
├── scripts/                    # Utility and run scripts
├── models/                     # Trained models and weights
├── results/                    # Evaluation outputs
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AAG1201/Audio_Declip.git
   cd Audio_Declip
   ```

2. **Create and Activate Environment**
   ```bash
   conda create -n aagproj python=3.9
   conda activate aagproj
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Dataset

Place your datasets as follows:

```
data/
├── train/
│   ├── audio1.wav
│   └── ...
├── test/
│   ├── audioA.wav
│   └── ...
├── pkl_data/
│   └── training_data.pkl
```

- You can either use existing `.pkl` files or generate new ones from `.wav` files using the script below.

---

## 🛠️ Generating Dataset from Custom .wav Files

To generate training data from your own `.wav` files:

```bash
python scripts/generate_dataset.py \
    --input_dir data/train \
    --output_pickle data/pkl_data/training_data.pkl \
    --frame_size 1024 \
    --hop_length 512
```

You can customize parameters like frame size and hop length depending on your model.

---

## 🧠 Training

Run the following to train the model:

```bash
python scripts/train_model.py \
    --data_pickle data/pkl_data/training_data.pkl \
    --epochs 100 \
    --batch_size 32 \
    --save_dir models/
```

---

## 🧪 Evaluation

Evaluate the trained model on test `.wav` files:

```bash
python scripts/evaluate_model.py \
    --model_path models/best_model.pth \
    --test_dir data/test \
    --output_dir results/
```

The declipped outputs and evaluation metrics (SNR, SDR, PESQ, etc.) will be saved in `results/`.

---

## ✍️ Notes

- The code supports **forward**, **backward**, and **bidirectional** training/evaluation.
- You can adjust training hyperparameters and model architecture from `config.py` or pass them as arguments.

---

## 📧 Contact

For questions, open an issue or reach out at [your_email@example.com].

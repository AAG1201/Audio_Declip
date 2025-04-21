
# Audio Declip Project

This repository provides code and tools for training and evaluating models for **audio declipping**. It supports using pre-generated datasets or generating datasets from custom `.wav` files.

---

## 📁 Repository Structure

```
Audio_Declip
├── aspade.py                       # Baseline ASPADE 
├── custom_output/                  # Evaluation outputs for custom inputs
├── custom_sound/                   # Your own .wav files for testing
├── custom_sound_variation/         # To find optimum window length for your audio dataset for training
├── dynamic_aspade.py               # Dynamic ASPADE
├── evaluate.py                     # Evaluation pipeline
├── loss_plots/                     # Training loss visualizations
├── main_ml.ipynb                   # Jupyter notebook for ML pipeline
├── ml_aspade.py                    # ML based ASPADE 
├── pipeline.py                     # Main data processing pipeline
├── pkl_data/
│   └── training_data.pkl           # Pickled training data
├── requirements.txt                # Python dependencies
├── saved_models/                   # Checkpoints and final models
├── spade_segmentation.py           # Bridge
├── test_data/                      # Test .wav files
├── toolbox/                        # Utility functions for audio processing
├── train_data/                     # Train .wav files
├── train_data_gen.py               # Dataset generation (pkl data)  from .wav
├── training.py                     # Main training script
├── variation_study.py              # Analysis on variation experiments on custom_sound_variations
└── variation_study_summary.xlsx
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

The dataset used for training and evaluation (including `train_data/`, `test_data/`, `pkl_data/`, and `saved_models/`) can be accessed via the following Google Drive link:

👉 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1qMC818ggFpiL7YZ8FpsZGSW3iVOSZmOu?usp=sharing)**

Place the downloaded folders as follows within the project directory:

```
Audio_Declip/
├── train_data
│   ├── audio1.wav
│   └── ...
├── test_data
│   ├── audioA.wav
│   └── ...
├── pkl_data/
│   └── training_data.pkl
├── saved_models/
```

- You can either use existing `.pkl` files or generate new ones from `.wav` files using the script below.

---

## 🛠️ Generating Dataset from Custom .wav Files

To generate training data from your own `.wav` files:

```bash
python train_data_gen.py --audio_dir "your_audio_path" \
    --cnt 2500 \
    --train_dir "train_data" \
    --test_dir "test_data" \
    --output_path "pkl_data" \
    --target_fs_values 16000 \
    --clipping_thresholds 0.1 0.2 \
    --time_clip 1 --win_len 500 \
    --win_shift 125 \
    --delta 300 \
    --s_ratio 0.9 
```

You can customize parameters like frame size and hop length depending on your model.

---

## 🧠 Training

Run the following to train the model:

```bash
python training.py --pkl_path pkl_data/training_data.pkl \
    --epochs 200 \
    --batch_size 1024 \
    --save_path saved_models \
    --plot_path loss_plots \
    --checkpoint_freq 10 \
    --resume 
```

---

## 🧪 Evaluation

Evaluate the trained model on test `.wav` files:

```bash
python evaluate.py --model_path "saved_models/final/complex_dft_unet_final.pth" \
    --test_audio_dir "custom_sound" \
    --output_dir "custom_output" \
    --target_fs_values 16000 \
    --clipping_thresholds 0.1 \
    --time_clip 1 \
    --factor 0.8 \
    --eval_mode 1 \
    --dynamic 0
```

The declipped outputs and evaluation metrics (SDR, PESQ, etc.) will be saved in `custom_output/`.

---

## ✍️ Notes

- The evaulation code ahs 3 modes - Baseline ASPADE, Dynamic ASPADE and ML based ASPADE
- For Baseline ASPADE set eval_mode to 0 and dynamic to 0
- For Dynamic ASPADE set eval_mode to 0 and dynamic to 1
- For ML absed ASPADE set eval_mode to 1

---

## 📧 Contact

For questions, open an issue or reach out at adityaag@iisc.ac.in.

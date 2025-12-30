# SRSC+: A Bilevel Sensitivity-Corrected Reconstruction Framework with Deep Priors for Parallel MRI

This repository provides the official implementation of **SRSC+**.

## 🔧 Requirements

The environment configuration is provided in [`environment.yml`](./environment.yml).
You can create the environment using:

```bash
conda env create -f environment.yml
```

## 🚀 Getting Started

To run the reconstruction on test data:

```bash
python main.py
```

The code includes a set of **Phantom data** for quick testing.

## 📁 Project Structure

* `main.py` – Entry point for running the reconstruction.
* `gen_mask.m` – MATLAB script for generating sampling masks.
* `environment.yml` – Conda environment specification.
* `data/` – Contains example Phantom test data.
* `mask/` – Contains example undersampled mode.
* `algorithm/SRSC.py` – Our propose ADDL algorithm.
* `utils/` – Supporting modules.

## 📦 Pretrained Models & Training Data & Undersampled Mode

Pretrained ADDL networks and additional test data are available at:

👉 [Google Drive - SRSC Resources](https://drive.google.com/drive/folders/1-VAxvTfvSRAaP-iesDgRdYz5vY1tLILD?usp=drive_link)

* Please place the file `net.pth` in folder [`utils/checkpoint`](./utils/checkpoint).
* Generate your own sampling patterns via `gen_mask.m`.

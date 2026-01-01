# LightMedVLM: A Lightweight Vision-Language Framework for High-Performance Radiology Report Generation

**Bachelor Thesis Implementation**  \
**Authors:** Duc Anh Vu, Huy Hoang Tran, Le Minh Vu Pham  \
**Supervisor:** Assoc. Prof. Duy Hung Phan  \
**Institution:** FPT University, Hoa Lac Campus  \
**Year:** 2025

---

## 📌 Overview

This repository contains the official implementation of **LightMedVLM**, a lightweight Multimodal Large Language Model (MLLM) designed for resource-constrained medical environments.

The framework integrates a **Swin Transformer** vision encoder with a **Qwen3-0.6B** Nano-LLM using a novel **Spatially-Aware Semantic Abstractor (SASA)**. Unlike traditional projectors that flood small models with redundant visual tokens, SASA functions as an active disease detector, utilizing convolutional texture refinement and knowledge-guided attention to maximize the signal-to-noise ratio. This architecture achieves high-performance radiology report generation and Visual Question Answering (VQA) on consumer-grade hardware.

---

## 📂 Repository Structure

```text
LightMedVLM/
├── processing_data/                # Scripts for data cleaning, HDBSCAN clustering, and balancing
├── scripts/                        # Utility scripts for setup and evaluation metrics
├── train/                          # Training scripts for the Baseline architecture
├── train_advanced/                 # Training scripts for the Advanced architecture (with SASA)
├── train_vqa_advanced/             # Fine-tuning scripts for the VQA task (Phase 3)
├── inference_report_generation.py  # Generate radiology reports from images
├── inference_vqa.py                # Visual Question Answering inference
├── model.py                        # Swin Transformer, SASA Connector, and Qwen LLM definition
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

---

## 🛠️ Installation

### Prerequisites

- **Python:** 3.10+
- **Hardware:** CUDA-enabled GPU (tested on NVIDIA T4 and RTX 3090)

### Setup Steps

Clone the repository:

```bash
git clone https://github.com/yourusername/LightMedVLM.git
cd LightMedVLM
```

Create a virtual environment:

```bash
conda create -n lightmedvlm python=3.10
conda activate lightmedvlm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Data Preparation

> **Note:** This project utilizes the **MIMIC-CXR** dataset. Access requires credentialing via PhysioNet.

### Preprocessing & Clustering

Navigate to the `processing_data` folder to clean raw MIMIC reports, extract CheXbert labels, and run HDBSCAN clustering to analyze data distribution.

```bash
cd processing_data
python run_clustering_pipeline.py \
  --input_path /path/to/mimic \
  --output_path ./processed_data
```

This process generates a cluster-based reduced dataset used in the **Advanced** training phases to mitigate class imbalance.

---

## 🚀 Training Pipeline

The training follows a **three-stage curriculum**.

### Stage 1 & 2: Feature Alignment & Report Generation

#### Option A: Baseline Model (Linear Projector)

```bash
python train/train_baseline.py --data_path /path/to/processed_data
```

#### Option B: Advanced Model (SASA Connector)

**Stage 1: Feature Alignment (Frozen LLM)**

```bash
python train_advanced/train_phase1_alignment.py \
  --config configs/sasa_config.yaml
```

**Stage 2: Report Generation Fine-tuning (Unfrozen LLM + LoRA)**

```bash
python train_advanced/train_phase2_finetune.py \
  --resume_from checkpoints/phase1.pth
```

### Stage 3: Visual Question Answering (VQA)

Fine-tunes the model on the VQA dataset mixed with synthetic minority-class questions to enhance diagnostic sensitivity.

```bash
python train_vqa_advanced/train_phase3_vqa.py \
  --resume_from checkpoints/phase2.pth
```

---

## ⚡ Inference

Standalone scripts are provided to evaluate the trained model on single images.

### 1. Radiology Report Generation

Generates full **Findings** and **Impression** sections for a chest X-ray image.

```bash
python inference_report_generation.py \
  --image_path samples/test_xray.jpg \
  --model_path checkpoints/lightmedvlm_best.pth \
  --output_file result.txt
```

### 2. Medical Visual Question Answering (VQA)

Answers natural language questions about the image (e.g., *"Is there consolidation?"*).

```bash
python inference_vqa.py \
  --image_path samples/test_xray.jpg \
  --question "Is there any sign of pleural effusion?" \
  --model_path checkpoints/lightmedvlm_best.pth
```

---

## 🧩 Model Architecture (`model.py`)

The core hybrid architecture consists of:

- **Vision Encoder:** Swin Transformer (Base), initialized with ImageNet weights
- **SASA Connector:**
  - **Stage A – C-Abstractor:** Convolutional refinement for local texture preservation
  - **Stage B – Query Attention:** Cross-attention initialized with 14 disease embeddings
- **LLM:** Qwen3-0.6B optimized with Low-Rank Adaptation (LoRA)

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🎓 Citation

If you use this code or methodology, please cite the thesis:

```bibtex
@thesis{LightMedVLM2025,
  title  = {LightMedVLM: A Lightweight Vision-Language Framework for High-Performance Radiology Report Generation},
  author = {Vu, Duc Anh and Tran, Huy Hoang and Pham, Le Minh Vu},
  school = {FPT University},
  year   = {2025}
}
```


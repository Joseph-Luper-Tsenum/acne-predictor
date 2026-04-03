# SkinCNN: CNN-Based Acne Detection from Dermatological Images

> A deep learning pipeline for binary skin lesion classification (acne vs. normal) using CNNs trained from scratch and via transfer learning, with Grad-CAM interpretability.

**BME6938: Medical AI · Project 2 · Group 7 · University of Florida · Spring 2026**

---

## Clinical Context

Acne vulgaris is the most common skin condition worldwide, affecting approximately 85% of adolescents and young adults. While typically diagnosed visually by dermatologists, access to specialist care is limited in many settings. Automated classification of skin lesions from dermatological images could support triage in primary care and enable remote dermatological screening. This project develops and evaluates CNN-based classifiers that distinguish acne lesions from normal skin, comparing models trained from scratch with pretrained architectures fine-tuned via transfer learning.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **ResNet-18** | **100%** | 1.000 | 1.000 | 1.000 | **1.000** |
| **DenseNet-121** | **100%** | 1.000 | 1.000 | 1.000 | **1.000** |
| **EfficientNet-B0** | **100%** | 1.000 | 1.000 | 1.000 | **1.000** |
| Custom CNN | 75.6% | 0.791 | 0.944 | 0.861 | 0.583 |

Transfer learning dramatically outperformed the from-scratch CNN, demonstrating the value of ImageNet-pretrained features for small medical image datasets.

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── run_training.sh                    # SLURM script for HiPerGator
│
├── notebooks/
│   └── Project2_SkinLesion_CNN.ipynb  # Main training & evaluation notebook
│
├── models/
│   ├── custom_cnn_best.pth
│   ├── resnet_18_best.pth             # (not uploaded due to size)
│   ├── densenet_121_best.pth
│   └── efficientnet_b0_best.pth
│
├── figures/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── augmented_samples.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── roc_pr_curves.png
│   ├── model_comparison.png
│   └── gradcam_results.png
│
├── logs/
│   ├── custom_cnn_history.csv
│   ├── resnet_18_history.csv
│   ├── densenet_121_history.csv
│   └── efficientnet_b0_history.csv
│
└── reports/
    └── Project2_Group6_Report.pdf
```

## Dataset

- **Source:** Skin lesion images provided on HiPerGator (`/blue/bme6938/share/Datasets/SkinLesions/`)
- **Size:** 300 images (244×244 RGB)
- **Classes:** Acne (240 images, 80%) vs. Normal (60 images, 20%)
- **Split:** 70% train (210) / 15% validation (45) / 15% test (45), stratified
- **Augmentation:** Random horizontal/vertical flip, rotation (30°), affine transform, ColorJitter, random grayscale

## Environment Setup

### On HiPerGator

```bash
module load conda
conda activate medai  # or your environment
pip install -r requirements.txt
```

### Local

```bash
pip install -r requirements.txt
```

**Computational requirements:** GPU recommended (A100 on HiPerGator). Training takes ~5 minutes for all 4 models.

## Quick Start

### Run the Notebook

```bash
jupyter notebook notebooks/Project2_SkinLesion_CNN.ipynb
```

### Run via SLURM (HiPerGator)

```bash
cd /blue/bme6938/Josephtsenum/Project2
sbatch run_training.sh
```

### Expected Outputs

After running the notebook, the following are generated:
- 4 trained model checkpoints in `models/`
- 8 figures in `figures/`
- 4 training history CSVs in `logs/`
- Test results table (`figures/test_results.csv`)

## Models

| Model | Type | Parameters | Epochs | Best Val Acc |
|-------|------|-----------|--------|-------------|
| Custom CNN | From scratch (3-block) | ~600K | 23 (early stop) | ~80% |
| ResNet-18 | Transfer learning | 11.2M | 18 (early stop) | 100% |
| DenseNet-121 | Transfer learning | 7.0M | 32 (early stop) | 100% |
| EfficientNet-B0 | Transfer learning | 4.0M | 38 (early stop) | 100% |

## Authors

**Joseph Luper Tsenum:** Ph.D. Researcher in Biomedical Engineering (Modeling & Biomedical Data Science Specialization), University of Florida. Joseph develops Generative AI platforms for designing novel oligonucleotides and applies machine learning methods to biomedical data analysis and drug discovery.

## Team Contributions

| Member | Contributions |
|--------|--------------|
| **Joseph Luper Tsenum** | Pipeline architecture, model training, transfer learning, Grad-CAM implementation, report writing, GitHub management |
| **Maria C. Horey** | Data exploration, augmentation strategy, literature review, evaluation analysis |
| **Benjamin D. Tondre** | Transfer learning setup, hyperparameter tuning, training dynamics analysis, report writing |

## Collaboration 

Throughout the project, the team maintained a highly collaborative workflow, meeting regularly to discuss progress, make decisions, and coordinate tasks. As a group, the team collectively shaped the direction of the project and worked together across all stages, including data preprocessing, model development, evaluation, report preparation, and application development. 

Meetings were held where team members jointly reviewed analyses, implemented modeling approaches, and refined the outputs for each phase. The final notebooks and project artifacts were compiled collaboratively to ensure consistency and reproducibility across the entire pipeline. 

Joseph was responsible for ensuring that the various components of the project—including data preprocessing, modeling outputs, explainability analyses, and any interactive deliverables remained aligned and coherent across phases. At the same time, the collaborative contributions of the entire team made it possible to efficiently develop supporting documentation and presentation materials, as different sections produced by team members were integrated into a unified narrative. 

This project reflects the type of collaborative environment commonly encountered in real-world industry and research settings, where interdisciplinary teams contribute complementary expertise. Team members were able to work together productively, resolve challenges constructively, and learn from one another's technical strengths and soft skills, resulting in a cohesive and well-executed final product. 

## References

1. He, K., et al. (2016). Deep Residual Learning. CVPR.
2. Huang, G., et al. (2017). Densely Connected CNNs. CVPR.
3. Tan, M., & Le, Q. (2019). EfficientNet. ICML.
4. Selvaraju, R., et al. (2017). Grad-CAM. ICCV.

---

⚠ **Disclaimer:** This tool is for educational purposes only and is not intended for clinical diagnosis.

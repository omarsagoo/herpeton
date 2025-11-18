## herpeton
<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1;">
    <img src="images/herpeton_logo.png" alt="Herpeton Logo" width="150">
  </div>
  <div style="flex: 2; text-align: right;">
    <p><strong>herpeton</strong> is a computer vision project focused on reptile detection, conservation, and ecological monitoring using deep learning techniques.</p>
  </div>
</div>
The Greek word herpeton (ἑρπετόν) means “creeping thing” or “reptile.”

It comes from the Greek verb herpein (ἕρπειν), which means “to creep” or “to crawl.”
The term reflects how these animals, such as snakes, lizards, and other reptiles, move close to the ground.

The modern word herpetology literally means “the study of creeping animals.”

# Automated Reptile Species Classification Using the BioTrove Dataset


## AAI-521: Applied Computer Vision for AI  
This project is a part of the AAI-521 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

**Project Status:** Active (In Progress)

# Installation

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Quick start with the dataset:**
   ```bash
   python quick_start.py
   ```

3. **Or use the Jupyter notebook for detailed exploration:**
   ```bash
   jupyter notebook biotrove_reptilia_loader.ipynb
   ```

## Detailed Installation

### Prerequisites
- Python 3.8 or higher
- At least 5GB free disk space (for full dataset)

### Step-by-step Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd herpeton

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install required packages
pip install -r requirements.txt

# Start with the dataset
python quick_start.py
```

# Project Introduction / Objective
The project develops a computer vision pipeline to identify **reptile species** such as snakes, lizards, turtles, and geckos from field imagery. Using the **BioTrove biodiversity dataset**, it explores how **CNNs**, **Vision Transformers (ViT)**, and **YOLOv10** can enhance conservation and biodiversity monitoring.

# Partner(s) / Contributor(s)
- Carrie Little, clittle@sandiego.edu
- Dean P. Simmer, dsimmer@sandiego.edu
- Omar Sagoo, osagoo@sandiego.edu

# Methods Used
- Deep Learning (CNNs, Vision Transformers)  
- Object Detection (YOLOv10 / Ultralytics)  
- Transfer Learning (ImageNet & BioTrove-CLIP)  
- Data Augmentation and Preprocessing  
- Model Evaluation & Visualization (Grad-CAM, Confusion Matrices)  
- Ethics for AI and Responsible Wildlife Applications  

# Technologies
- Python
- Google Colab / Jupyter Notebook  
- Hugging Face Datasets
- PyTorch / Ultralytics YOLOv10 
- NumPy, Pandas, Matplotlib, Seaborn
- ImageHash
- OpenCV, PIL
- TensorBoard

# Project Description

This project leverages the **BioTrove-TRAIN Reptilia subset** from the **BioTrove dataset** (Hugging Face), containing ~1.3 million labeled reptile images across 189 species.  
Images range from 224–1024 px and include taxonomy metadata (class, family, genus, species, scientific name, location).

## Dataset Information

### BioTrove-TRAIN Reptilia Subset
- **Source**: [BGLab/BioTrove-Train](https://huggingface.co/datasets/BGLab/BioTrove-Train) on Hugging Face
- **Size**: ~1.3M reptile images from ~40M total biodiversity images
- **Species Coverage**: 189+ reptile species across major families
- **Image Resolution**: 224-1024 pixels
- **Format**: RGB images with comprehensive metadata

### Key Features
- **Taxonomic Hierarchy**: Full classification (Kingdom → Species)
- **Geographic Distribution**: Global coverage with location metadata
- **Species Diversity**: Snakes, lizards, turtles, geckos, and more
- **High Quality**: Curated from iNaturalist with expert validation
- **Metadata Rich**: Scientific names, common names, family information

### Major Reptile Families Included
- **Viperidae** (Vipers and Pit Vipers)
- **Colubridae** (Colubrids - largest snake family)
- **Elapidae** (Venomous snakes including cobras, mambas)
- **Gekkonidae** (Geckos)
- **Iguanidae** (Iguanas and related lizards)
- **Testudinidae** (Tortoises)
- **Cheloniidae** (Sea turtles)
- And many more...

### Usage in Computer Vision
This dataset is specifically designed for:
- **Species Classification**: Multi-class reptile identification
- **Hierarchical Classification**: Family/genus/species level predictions
- **Transfer Learning**: Pre-trained on biodiversity data
- **Object Detection**: Bounding box annotations available
- **Conservation Applications**: Real-world wildlife monitoring


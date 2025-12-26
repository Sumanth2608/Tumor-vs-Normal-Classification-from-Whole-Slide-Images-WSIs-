# Tumor vs Normal Classification from Whole-Slide Images (WSIs)

This project aims to develop a deep learning model that classifies tissue regions as Tumor or Normal from Hematoxylin and Eosin (H&E)-stained Whole-Slide Images (WSIs). The model will also highlight tumor regions of interest (ROIs) to assist pathologists in diagnosis.

## Stage 1: Data Acquisition

Whole-Slide Images (WSIs) and corresponding labels will be collected from open-source datasets such as CAMELYON16/17 and The Cancer Genome Atlas (TCGA). The datasets contain H&E-stained slides with tumor and normal tissue annotations. Data will be organized in a structured directory format for efficient processing.

## Datasets

### CAMELYON16
- **Source**: https://camelyon16.grand-challenge.org/
- **Content**: 400 WSIs (270 training, 130 testing) with pixel-level tumor annotations
- **Format**: SVS files with XML annotations
- **Registration Required**: Yes, create account to download

### CAMELYON17
- **Source**: https://camelyon17.grand-challenge.org/
- **Content**: 1000 WSIs (500 training, 500 testing) with pixel-level tumor annotations
- **Format**: SVS files with XML annotations
- **Registration Required**: Yes, create account to download

### TCGA (The Cancer Genome Atlas)
- **Source**: https://portal.gdc.cancer.gov/
- **Content**: Thousands of WSIs from various cancer types (BRCA, LUAD, etc.)
- **Format**: SVS files
- **Registration Required**: Yes, NIH account recommended

## Data Structure

```
data/
├── raw/
│   ├── camelyon16/
│   │   ├── training/          # Training WSIs (.svs files)
│   │   ├── testing/           # Testing WSIs (.svs files)
│   │   └── lesion_annotations/# XML annotation files
│   ├── camelyon17/
│   │   ├── training/          # Training WSIs
│   │   └── testing/           # Testing WSIs
│   └── tcga/
│       ├── brca/              # Breast cancer WSIs
│       └── luad/              # Lung cancer WSIs
├── processed/                 # Processed patches (224x224)
│   ├── train/
│   │   ├── tumor/
│   │   └── normal/
│   └── val/
│       ├── tumor/
│       └── normal/
└── annotations/               # Processed annotation masks
```

## Setup

1. Clone or download the project.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. **Install OpenSlide**: Required for reading SVS files
   - Download from: https://openslide.org/download/
   - Install the Windows MSI installer
   - Then install Python bindings: `pip install openslide-python`

## Data Download Instructions

### Option 1: Use the Download Script
```bash
python download_data.py
```

### Option 2: Manual Download

#### CAMELYON16/17:
1. Visit the respective challenge websites
2. Register for an account
3. Download training and testing datasets
4. Extract to `data/raw/camelyon16/` or `data/raw/camelyon17/`

#### TCGA:
1. Visit https://portal.gdc.cancer.gov/
2. Search for cancer type (e.g., "TCGA-BRCA")
3. Filter by "Slide Image" data type
4. Select SVS format files
5. Download and extract to `data/raw/tcga/`

### Option 3: GDC Data Transfer Tool (Recommended for Large Downloads)
```bash
# Install GDC client
# Download from: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

# Download using manifest
gdc-client download -m manifest.txt
```

## Preprocessing

After downloading the data, preprocess the WSIs into patches:

```bash
python src/preprocess_data.py
```

This will:
- Extract 224×224 patches from WSIs
- Separate tumor and normal regions based on annotations
- Save processed patches to `data/processed/`

## Training

Train the ResNet-18 model:

```bash
python src/train.py
```

## Model Architecture

- **Base Model**: Pretrained ResNet-18
- **Input Size**: 224×224 RGB patches
- **Output**: Binary classification (Tumor/Normal)
- **Training**: Transfer learning with optional fine-tuning

## Requirements

- Python 3.8+
- PyTorch 1.9+
- TorchVision
- OpenSlide Python
- OpenSlide C library (system installation)
- Libraries listed in requirements.txt

## Usage

1. Download datasets using instructions above
2. Preprocess data: `python src/preprocess_data.py`
3. Train model: `python src/train.py`
4. Evaluate results and visualize predictions

## Notes

- CAMELYON datasets require registration and acceptance of terms
- TCGA data access may require NIH account for controlled access data
- Processing large WSIs requires significant storage space
- GPU recommended for training

## License

This project uses open-source datasets. Please refer to individual dataset licenses for usage terms.
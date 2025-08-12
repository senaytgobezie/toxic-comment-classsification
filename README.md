# Toxic Comment Classification

This project implements a multi-label text classification model for detecting six types of toxicity in online comments:
- **toxic**
- **severe_toxic**
- **obscene**
- **threat**
- **insult**
- **identity_hate**

The implementation uses transformer-based models (DeBERTa) to classify comments from the Jigsaw Toxic Comment Classification Challenge dataset.

## 🚨 Important: Download Required Dataset

**This project requires the Jigsaw Toxic Comment Classification Challenge dataset, which is NOT included in this repository due to size constraints.**

### How to Download the Dataset:

1. **Visit the Kaggle Competition Page**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

2. **Download the Required Files**:
   - `train.csv` - Training data with comments and toxicity labels (~159MB)
   - `test.csv` - Test data with comments (~63MB)
   - `test_labels.csv` - Test data labels (~1.5MB)
   - `sample_submission.csv` - Sample submission format (~1.5MB)

3. **Place the Files in Your Project Root Directory**:
   ```
   NLP/
   ├── train.csv
   ├── test.csv
   ├── test_labels.csv
   ├── sample_submission.csv
   └── ... (other project files)
   ```

4. **Note**: You'll need to accept the competition rules on Kaggle to download the dataset.

## 📁 Project Structure

```
NLP/
├── 📊 Data Files (Download from Kaggle)
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   ├── test_labels.csv        # Test labels
│   └── sample_submission.csv  # Submission format
├── 🐍 Core Python Files
│   ├── toxic_comment_classification.py           # Basic implementation
│   ├── toxic_comment_classification_enhanced.py  # Enhanced version with preprocessing
│   ├── text_preprocessing.py                     # Text preprocessing utilities
│   ├── focal_loss.py                            # Focal Loss implementation
│   ├── model_evaluation.py                      # Model evaluation scripts
│   └── quick_train.py                           # Quick training script
├── 📋 Configuration
│   ├── requirements.txt                          # Python dependencies
│   └── .gitignore                               # Git ignore rules
├── 📈 Outputs & Visualizations
│   ├── model_outputs/                            # Model checkpoints & results
│   ├── *.png                                     # Generated plots & charts
│   └── preprocessed_train.csv                    # Preprocessed training data
└── 🚫 Ignored Files (See explanation below)
    ├── venv/                                     # Virtual environment
    ├── toxic_env/                                # Conda environment
    ├── __pycache__/                              # Python cache
    └── *.pyc                                     # Compiled Python files
```

## 🚫 Ignored Files Explanation

The following files and directories are intentionally ignored by Git and are NOT included in this repository:

### **Virtual Environments** (venv/, toxic_env/)
- **Why Ignored**: These contain platform-specific Python packages and can be large
- **What to Do**: Create your own virtual environment using the provided `requirements.txt`

### **Dataset Files** (train.csv, test.csv, etc.)
- **Why Ignored**: Large file sizes (>200MB total) exceed GitHub's recommended limits
- **What to Do**: Download from [Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

### **Generated Files** (__pycache__/, *.pyc, *.png)
- **Why Ignored**: These are automatically generated and can be recreated
- **What to Do**: Run the scripts to generate these files locally

### **Model Outputs** (model_outputs/)
- **Why Ignored**: Contains large model files and training artifacts
- **What to Do**: Train the model locally to generate these files

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Kaggle account (for dataset download)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/senaytgobezie/toxic-comment-classsification.git
   cd toxic-comment-classsification
   ```

2. **Create Virtual Environment**
   ```bash
   # Using venv (recommended)
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset** (See section above)

5. **Verify Setup**
   ```bash
   python -c "import torch, transformers, pandas; print('Setup complete!')"
   ```

## 🚀 Running the Code

### Quick Start
```bash
# Basic model training
python toxic_comment_classification.py

# Enhanced model with preprocessing
python toxic_comment_classification_enhanced.py

# Text preprocessing and visualization
python text_preprocessing.py

# Quick training script
python quick_train.py
```

### Model Training Options

| Script | Features | Use Case |
|--------|----------|----------|
| `toxic_comment_classification.py` | Basic DeBERTa model | Quick experimentation |
| `toxic_comment_classification_enhanced.py` | Preprocessing + Focal Loss | Production-ready model |
| `text_preprocessing.py` | Data analysis + visualization | Understanding your data |
| `quick_train.py` | Streamlined training | Fast iteration |

## 🏗️ Model Architecture

- **Base Model**: DeBERTa-v3-small (configurable)
- **Text Processing**: Advanced cleaning, tokenization, augmentation
- **Loss Functions**: 
  - Binary Cross-Entropy (default)
  - Focal Loss (handles class imbalance)
  - Class-Balanced Loss (advanced balancing)
- **Training**: Automatic GPU/CPU detection, mixed precision support

## 📊 Data Preprocessing Features

- **Text Cleaning**: URL removal, HTML tag removal, special character handling
- **Data Augmentation**: Random deletion, word swapping, synonym replacement
- **Class Balancing**: Oversampling techniques for minority classes
- **Visualization**: Label distribution, comment length analysis, word clouds

## 🎯 Performance Metrics

The model evaluates performance using:
- **Macro F1-Score**: Overall performance across all classes
- **ROC AUC**: Per-class discrimination ability
- **Precision/Recall**: Detailed class-specific metrics
- **Confusion Matrices**: Visual error analysis

## ⚙️ Customization

### Model Parameters
```python
# In the script files, modify these variables:
MODEL_NAME = "microsoft/deberta-v3-small"  # Change base model
MAX_LENGTH = 512                           # Sequence length
BATCH_SIZE = 16                            # Training batch size
LEARNING_RATE = 2e-5                       # Learning rate
EPOCHS = 3                                 # Training epochs
```

### Feature Toggles
```python
USE_FOCAL_LOSS = True           # Enable Focal Loss
USE_PREPROCESSING = True        # Enable text preprocessing
USE_DATA_AUGMENTATION = True    # Enable data augmentation
```

## 🔧 Troubleshooting

### Common Issues

1. **"File not found: train.csv"**
   - Solution: Download dataset from Kaggle (see section above)

2. **CUDA out of memory**
   - Solution: Reduce `BATCH_SIZE` or use smaller model

3. **Import errors**
   - Solution: Ensure virtual environment is activated and dependencies installed

4. **Dataset loading issues**
   - Solution: Check file paths and ensure CSV files are in project root

### Performance Tips

- **GPU Users**: Increase batch size for faster training
- **CPU Users**: Use smaller models like "distilbert-base-uncased"
- **Memory Issues**: Reduce `MAX_LENGTH` or use gradient accumulation

## 📈 Expected Results

After running the enhanced model, you should see:
- **Training Progress**: Loss curves and metric updates
- **Model Checkpoint**: Saved in `model_outputs/` directory
- **Visualizations**: Multiple PNG files showing data analysis
- **Predictions**: CSV file ready for Kaggle submission

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Jigsaw/Conversation AI**: For the dataset and competition
- **Hugging Face**: For the transformers library
- **PyTorch**: For the deep learning framework

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the code comments for guidance
3. Open an issue on GitHub with detailed error information

---

**Remember**: This is a learning project for toxic comment classification. Always use AI responsibly and ethically! 
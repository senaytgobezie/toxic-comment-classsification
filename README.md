# Toxic Comment Classification

This project implements a multi-label text classification model for detecting six types of toxicity in online comments:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The implementation uses transformer-based models (DeBERTa) to classify comments from the Jigsaw Toxic Comment Classification Challenge dataset.

## Project Structure

- `toxic_comment_classification.py`: Basic implementation of the classification model
- `toxic_comment_classification_enhanced.py`: Enhanced version with preprocessing and focal loss
- `text_preprocessing.py`: Text preprocessing and data augmentation utilities
- `focal_loss.py`: Implementation of Focal Loss and Class-Balanced Loss for handling class imbalance
- `requirements.txt`: Required Python packages

## Dataset

The project uses the Jigsaw Toxic Comment Classification Challenge dataset, which consists of:
- `train.csv`: Training data with comments and toxicity labels
- `test.csv`: Test data with comments
- `test_labels.csv`: Test data labels
- `sample_submission.csv`: Sample submission format

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Code

### Basic Model

```
python toxic_comment_classification.py
```

### Enhanced Model with Preprocessing and Focal Loss

```
python toxic_comment_classification_enhanced.py
```

### Text Preprocessing Only

```
python text_preprocessing.py
```

## Model Details

- **Architecture**: DeBERTa-v3-small (can be changed in the code)
- **Text Preprocessing**: Cleaning, tokenization, and optional stopword removal
- **Data Augmentation**: Random deletion, random swap
- **Loss Functions**: Binary Cross-Entropy Loss, Focal Loss, Class-Balanced Loss
- **Evaluation Metrics**: Macro F1-score, ROC AUC, per-class metrics

## Results

The model outputs the following:
- Trained model checkpoint
- Submission file with predictions
- Training history plots
- Confusion matrices for each toxicity type
- Detailed metrics report

All outputs are saved in the `model_outputs` directory.

## Customization

You can customize the model by modifying the following parameters in the code:
- `MODEL_NAME`: Change the transformer model (e.g., "bert-base-uncased", "roberta-base")
- `MAX_LENGTH`: Maximum sequence length for tokenization
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for optimization
- `EPOCHS`: Number of training epochs
- `USE_FOCAL_LOSS`: Whether to use Focal Loss for handling class imbalance
- `USE_PREPROCESSING`: Whether to preprocess the text data
- `USE_DATA_AUGMENTATION`: Whether to use data augmentation techniques

## Performance Considerations

- The code is designed to run on either CPU or GPU, automatically detecting available hardware
- For large datasets, consider increasing batch size if using a GPU with sufficient memory
- For faster training, you can use a smaller model like "distilbert-base-uncased" 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from datetime import datetime
from tqdm import tqdm

# Import the ToxicCommentsDataset class from the toxic_comment_classification.py file
from toxic_comment_classification import ToxicCommentsDataset

# Define paths
TEST_LABELS_PATH = 'test_labels.csv/test_labels.csv'
OUTPUT_DIR = 'model_outputs'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'evaluation_results')

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(model_path, model_name="roberta-base"):
    """
    Load a saved model from checkpoint
    """
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Define label columns
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Load model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_columns),
        problem_type="multi_label_classification"
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, test_dataloader, threshold=0.5):
    """
    Evaluate the model on the test set and return predictions and true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_preds_binary = (all_preds > threshold).astype(int)
    
    if all_labels:
        all_labels = np.array(all_labels)
        return all_preds, all_preds_binary, all_labels
    else:
        return all_preds, all_preds_binary, None

def plot_confusion_matrices(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrices for each class
    """
    plt.figure(figsize=(20, 15))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(2, 3, i+1)
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        # Create a DataFrame for better visualization
        cm_df = pd.DataFrame(cm, 
                            index=['Actual Negative', 'Actual Positive'],
                            columns=['Predicted Negative', 'Predicted Positive'])
        
        # Plot heatmap
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {class_name}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """
    Plot ROC curves for each class
    """
    plt.figure(figsize=(12, 10))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14)
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curves(y_true, y_pred_proba, class_names, save_path):
    """
    Plot precision-recall curves for each class
    """
    plt.figure(figsize=(12, 10))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_threshold_impact(y_true, y_pred_proba, class_names, save_path):
    """
    Plot the impact of different thresholds on precision, recall, and F1-score
    """
    thresholds = np.arange(0.1, 1.0, 0.1)
    metrics = {class_name: {'precision': [], 'recall': [], 'f1': []} for class_name in class_names}
    
    # Calculate metrics for different thresholds
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        for i, class_name in enumerate(class_names):
            precision = precision_score(y_true[:, i], y_pred[:, i])
            recall = recall_score(y_true[:, i], y_pred[:, i])
            f1 = f1_score(y_true[:, i], y_pred[:, i])
            
            metrics[class_name]['precision'].append(precision)
            metrics[class_name]['recall'].append(recall)
            metrics[class_name]['f1'].append(f1)
    
    # Plot the results
    plt.figure(figsize=(20, 15))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(2, 3, i+1)
        
        plt.plot(thresholds, metrics[class_name]['precision'], 'b-', label='Precision')
        plt.plot(thresholds, metrics[class_name]['recall'], 'g-', label='Recall')
        plt.plot(thresholds, metrics[class_name]['f1'], 'r-', label='F1-score')
        
        plt.title(f'Metrics vs Threshold for {class_name}', fontsize=14)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Define label columns
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create a timestamp for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find the latest model checkpoint
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('model_checkpoint_') and f.endswith('.pt')]
    if not model_files:
        print("No model checkpoint found. Please train the model first.")
        return
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(OUTPUT_DIR, latest_model)
    print(f"Using model checkpoint: {model_path}")
    
    # Load the model
    model = load_model(model_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Load test data with labels
    test_labels_df = pd.read_csv(TEST_LABELS_PATH)
    
    # Prepare test dataset
    test_dataset = ToxicCommentsDataset(
        test_labels_df['comment_text'].values,
        test_labels_df[label_columns].values,
        tokenizer,
        max_length=256
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32
    )
    
    # Evaluate the model
    print("Evaluating model...")
    predictions, predictions_binary, true_labels = evaluate_model(model, test_dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels.flatten(), predictions_binary.flatten())
    precision_macro = precision_score(true_labels, predictions_binary, average='macro')
    recall_macro = recall_score(true_labels, predictions_binary, average='macro')
    f1_macro = f1_score(true_labels, predictions_binary, average='macro')
    roc_auc = roc_auc_score(true_labels, predictions, average='macro')
    
    # Per-class metrics
    class_metrics = {}
    for i, label in enumerate(label_columns):
        class_metrics[label] = {
            'accuracy': accuracy_score(true_labels[:, i], predictions_binary[:, i]),
            'precision': precision_score(true_labels[:, i], predictions_binary[:, i]),
            'recall': recall_score(true_labels[:, i], predictions_binary[:, i]),
            'f1': f1_score(true_labels[:, i], predictions_binary[:, i]),
            'roc_auc': roc_auc_score(true_labels[:, i], predictions[:, i])
        }
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"ROC AUC (Macro): {roc_auc:.4f}")
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    for label, metrics in class_metrics.items():
        print(f"\n{label.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrices
    plot_confusion_matrices(
        true_labels, 
        predictions_binary, 
        label_columns, 
        os.path.join(RESULTS_DIR, f'confusion_matrices_{timestamp}.png')
    )
    
    # ROC curves
    plot_roc_curves(
        true_labels, 
        predictions, 
        label_columns, 
        os.path.join(RESULTS_DIR, f'roc_curves_{timestamp}.png')
    )
    
    # Precision-recall curves
    plot_precision_recall_curves(
        true_labels, 
        predictions, 
        label_columns, 
        os.path.join(RESULTS_DIR, f'pr_curves_{timestamp}.png')
    )
    
    # Threshold impact
    plot_threshold_impact(
        true_labels, 
        predictions, 
        label_columns, 
        os.path.join(RESULTS_DIR, f'threshold_impact_{timestamp}.png')
    )
    
    # Save results to JSON
    results = {
        'timestamp': timestamp,
        'model_checkpoint': latest_model,
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'roc_auc_macro': float(roc_auc)
        },
        'class_metrics': {label: {k: float(v) for k, v in metrics.items()} for label, metrics in class_metrics.items()}
    }
    
    with open(os.path.join(RESULTS_DIR, f'evaluation_results_{timestamp}.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation complete. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main() 
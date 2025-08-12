import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
import os
import time
from datetime import datetime

# Import custom modules
from text_preprocessing import preprocess_dataset, clean_text
from focal_loss import FocalLoss, ClassBalancedLoss

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
TRAIN_PATH = 'train.csv/train.csv'
TEST_PATH = 'test.csv/test.csv'
TEST_LABELS_PATH = 'test_labels.csv/test_labels.csv'
SAMPLE_SUBMISSION_PATH = 'sample_submission.csv/sample_submission.csv'
OUTPUT_DIR = 'model_outputs'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Dataset
class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

# Model training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs, device, model_save_path):
    best_val_f1 = 0
    best_model_state = None
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            # Compute loss
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                
                # Compute loss
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Calculate metrics
        train_f1_macro = f1_score(
            (train_labels > 0.5).astype(int), 
            (train_preds > 0.5).astype(int), 
            average='macro'
        )
        
        val_f1_macro = f1_score(
            (val_labels > 0.5).astype(int), 
            (val_preds > 0.5).astype(int), 
            average='macro'
        )
        
        val_roc_auc = roc_auc_score(val_labels, val_preds, average='macro')
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_f1'].append(train_f1_macro)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1_macro)
        history['val_auc'].append(val_roc_auc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Train F1 Macro: {train_f1_macro:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val F1 Macro: {val_f1_macro:.4f}, Val ROC AUC: {val_roc_auc:.4f}")
        
        # Save best model
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model_state = model.state_dict().copy()
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1_macro,
                'val_auc': val_roc_auc
            }, model_save_path)
            
            print(f"New best model saved with Val F1 Macro: {best_val_f1:.4f}")
            
            # Per-class metrics
            val_preds_binary = (val_preds > 0.5).astype(int)
            val_labels_binary = (val_labels > 0.5).astype(int)
            
            print("\nPer-class validation metrics:")
            for i, label in enumerate(label_columns):
                class_f1 = f1_score(val_labels_binary[:, i], val_preds_binary[:, i])
                class_auc = roc_auc_score(val_labels[:, i], val_preds[:, i])
                print(f"{label}: F1 = {class_f1:.4f}, AUC = {class_auc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.title('Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    
    return best_model_state, history

# Main execution
def main():
    # Model parameters
    MODEL_NAME = "roberta-base"  # Changed from microsoft/deberta-v3-small
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    USE_FOCAL_LOSS = True
    USE_PREPROCESSING = True
    USE_DATA_AUGMENTATION = True
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(OUTPUT_DIR, f"model_checkpoint_{timestamp}.pt")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_labels_df = pd.read_csv(TEST_LABELS_PATH)
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Explore data
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Test labels shape: {test_labels_df.shape}")
    
    # Check columns
    print("\nTrain columns:")
    print(train_df.columns.tolist())
    
    # Define label columns
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Check label distribution
    print("\nLabel distribution in training set:")
    for col in label_columns:
        print(f"{col}: {train_df[col].sum()} ({train_df[col].mean()*100:.2f}%)")
    
    # Preprocess data if enabled
    if USE_PREPROCESSING:
        print("\nPreprocessing data...")
        train_df = preprocess_dataset(train_df, augment=USE_DATA_AUGMENTATION)
        test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
        
        # Check new label distribution after preprocessing
        print("\nLabel distribution after preprocessing:")
        for col in label_columns:
            print(f"{col}: {train_df[col].sum()} ({train_df[col].mean()*100:.2f}%)")
    
    # Create a validation set
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED, stratify=train_df['toxic'])
    print(f"\nTraining set: {train_df.shape[0]} examples")
    print(f"Validation set: {val_df.shape[0]} examples")
    
    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_columns),
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = ToxicCommentsDataset(
        train_df['comment_text'].values,
        train_df[label_columns].values,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = ToxicCommentsDataset(
        val_df['comment_text'].values,
        val_df[label_columns].values,
        tokenizer,
        MAX_LENGTH
    )
    
    test_dataset = ToxicCommentsDataset(
        test_df['comment_text'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Set up loss function
    if USE_FOCAL_LOSS:
        print("\nUsing Focal Loss...")
        # Calculate class weights based on training data
        class_counts = train_df[label_columns].sum().values
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        print("\nUsing BCE Loss...")
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train model
    print("\nTraining model...")
    best_model_state, history = train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        criterion,
        EPOCHS,
        device,
        model_save_path
    )
    
    # Load best model for inference
    model.load_state_dict(best_model_state)
    
    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(preds)
    
    test_preds = np.array(test_preds)
    
    # Create submission file
    print("\nCreating submission file...")
    submission_df = sample_submission_df.copy()
    submission_df[label_columns] = test_preds
    submission_path = os.path.join(OUTPUT_DIR, f"submission_{timestamp}.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created: {submission_path}")
    
    # Evaluate on test labels if available
    if 'toxic' in test_labels_df.columns:
        print("\nEvaluating on test set...")
        test_labels = test_labels_df[label_columns].values
        test_preds_binary = (test_preds > 0.5).astype(int)
        
        test_f1_macro = f1_score(test_labels, test_preds_binary, average='macro')
        test_roc_auc = roc_auc_score(test_labels, test_preds, average='macro')
        
        print(f"Test F1 Macro: {test_f1_macro:.4f}, Test ROC AUC: {test_roc_auc:.4f}")
        
        print("\nPer-class test metrics:")
        class_metrics = {}
        for i, label in enumerate(label_columns):
            class_f1 = f1_score(test_labels[:, i], test_preds_binary[:, i])
            class_auc = roc_auc_score(test_labels[:, i], test_preds[:, i])
            print(f"{label}: F1 = {class_f1:.4f}, AUC = {class_auc:.4f}")
            class_metrics[label] = {'f1': class_f1, 'auc': class_auc}
        
        # Save metrics to file
        metrics_df = pd.DataFrame(class_metrics).T
        metrics_df['overall_f1'] = test_f1_macro
        metrics_df['overall_auc'] = test_roc_auc
        metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_path)
        print(f"Metrics saved to {metrics_path}")
        
        # Plot confusion matrices for each class
        plt.figure(figsize=(15, 10))
        for i, label in enumerate(label_columns):
            plt.subplot(2, 3, i+1)
            cm = pd.crosstab(
                pd.Series(test_labels[:, i], name='Actual'),
                pd.Series(test_preds_binary[:, i], name='Predicted')
            )
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {label}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrices_{timestamp}.png"))
        plt.close()

if __name__ == "__main__":
    main() 
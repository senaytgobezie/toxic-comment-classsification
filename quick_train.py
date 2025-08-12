import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import os
from datetime import datetime

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
OUTPUT_DIR = 'model_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define label columns
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Dataset class
class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
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

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            predictions.extend(preds)
            
            actual_labels.extend(batch['labels'].cpu().numpy())
    
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    # Calculate metrics
    roc_auc = roc_auc_score(actual_labels, predictions, average='macro')
    f1 = f1_score((predictions >= 0.5).astype(int), actual_labels, average='macro')
    
    return {
        'roc_auc': roc_auc,
        'f1_score': f1
    }

def main():
    # Model parameters
    MODEL_NAME = "distilroberta-base"  # Smaller model than roberta-base
    MAX_LENGTH = 128  # Reduced from 256
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 1  # Reduced from 3
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(TRAIN_PATH)
    
    # Take a very small subset of the data (1% for faster training)
    df = df.sample(frac=0.01, random_state=RANDOM_SEED)
    print(f"Using {len(df)} samples for training")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    
    print(f"Training set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    
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
        texts=train_df['comment_text'].values,
        labels=train_df[label_columns].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    val_dataset = ToxicCommentsDataset(
        texts=val_df['comment_text'].values,
        labels=val_df[label_columns].values,
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
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_auc = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_dataloader, device)
        print(f"Validation ROC AUC: {val_metrics['roc_auc']:.4f}")
        print(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")
        
        # Save model if it's the best so far
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            model_save_path = os.path.join(OUTPUT_DIR, f"quick_model_best.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 
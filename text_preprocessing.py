import re
import random
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text cleaning function
def clean_text(text):
    """
    Clean text by removing special characters, URLs, and extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Remove stopwords
def remove_stopwords(text):
    """
    Remove stopwords from text
    """
    if not isinstance(text, str) or not text:
        return ""
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

# Data augmentation techniques
def random_deletion(text, p=0.1):
    """
    Randomly delete words from text with probability p
    """
    if not isinstance(text, str) or not text:
        return ""
    
    words = text.split()
    if len(words) <= 1:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    if len(new_words) == 0:
        # If all words were deleted, keep a random one
        return random.choice(words)
    
    return ' '.join(new_words)

def random_swap(text, n=1):
    """
    Randomly swap n pairs of words in the text
    """
    if not isinstance(text, str) or not text:
        return ""
    
    words = text.split()
    if len(words) <= 1:
        return text
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words)

def synonym_replacement(text, n=1):
    """
    Replace n words in the text with their synonyms
    Note: This is a simplified version without actual synonym replacement
    For a real implementation, you'd need a synonym dictionary or WordNet
    """

    return text

# Handle class imbalance
def oversample_minority_classes(df, label_columns, random_state=42):
    """
    Oversample minority classes to balance the dataset
    """
    # Create a new DataFrame to store the balanced dataset
    balanced_df = df.copy()
    
    # For each label column
    for col in label_columns:
        # Get the majority and minority classes
        majority_class = df[df[col] == 0]
        minority_class = df[df[col] == 1]
        
        # If minority class is less than 30% of majority class, oversample
        if len(minority_class) < len(majority_class) * 0.3:
            # Calculate how many samples to generate
            n_samples = int(len(majority_class) * 0.3)
            
            # Oversample the minority class
            minority_oversampled = resample(
                minority_class,
                replace=True,
                n_samples=n_samples,
                random_state=random_state
            )
            
            # Combine the oversampled minority class with the original data
            balanced_df = pd.concat([balanced_df, minority_oversampled])
    
    return balanced_df

# Apply all preprocessing steps
def preprocess_dataset(df, text_column='comment_text', clean=True, remove_stop=False, augment=False):
    """
    Apply all preprocessing steps to the dataset
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean text
    if clean:
        processed_df[text_column] = processed_df[text_column].apply(clean_text)
    
    # Remove stopwords
    if remove_stop:
        processed_df[text_column] = processed_df[text_column].apply(remove_stopwords)
    
    # Data augmentation
    if augment:
        # Create augmented samples for toxic comments
        augmented_samples = []
        
        # Define label columns
        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Filter toxic comments (any label is 1)
        toxic_df = processed_df[processed_df[label_columns].sum(axis=1) > 0]
        
        # For each toxic comment, create augmented versions
        for _, row in toxic_df.iterrows():
            # Create augmented text with random deletion
            aug_text1 = random_deletion(row[text_column])
            augmented_samples.append({
                text_column: aug_text1,
                **{col: row[col] for col in label_columns}
            })
            
            # Create augmented text with random swap
            aug_text2 = random_swap(row[text_column])
            augmented_samples.append({
                text_column: aug_text2,
                **{col: row[col] for col in label_columns}
            })
        
        # Convert augmented samples to DataFrame and append to processed_df
        if augmented_samples:
            aug_df = pd.DataFrame(augmented_samples)
            processed_df = pd.concat([processed_df, aug_df], ignore_index=True)
    
    return processed_df

# Visualization functions
def plot_label_distribution(df, label_columns):
    """
    Plot the distribution of labels in the dataset
    """
    plt.figure(figsize=(12, 6))
    
    # Count of each label
    label_counts = df[label_columns].sum().sort_values(ascending=False)
    
    # Create bar plot
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Distribution of Toxic Comment Labels', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add percentage labels on top of bars
    total = len(df)
    for i, count in enumerate(label_counts.values):
        percentage = count / total * 100
        plt.text(i, count + 50, f'{percentage:.1f}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.show()

def plot_comment_length_distribution(df, text_column='comment_text'):
    """
    Plot the distribution of comment lengths
    """
    # Calculate comment lengths
    df['comment_length'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of comment lengths
    sns.histplot(data=df, x='comment_length', bins=50, kde=True)
    plt.title('Distribution of Comment Lengths', fontsize=16)
    plt.xlabel('Comment Length (words)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xlim(0, 200)  # Focus on comments with less than 200 words
    
    plt.tight_layout()
    plt.savefig('comment_length_distribution.png')
    plt.show()
    
    # Compare toxic vs non-toxic comment lengths
    plt.figure(figsize=(12, 6))
    
    # Define toxic comments as those with any toxic label
    df['is_toxic'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0).astype(int)
    
    # Plot boxplot of comment lengths by toxicity
    sns.boxplot(x='is_toxic', y='comment_length', data=df)
    plt.title('Comment Length by Toxicity', fontsize=16)
    plt.xlabel('Is Toxic', fontsize=14)
    plt.ylabel('Comment Length (words)', fontsize=14)
    plt.xticks([0, 1], ['Non-Toxic', 'Toxic'])
    
    plt.tight_layout()
    plt.savefig('comment_length_by_toxicity.png')
    plt.show()

def generate_wordclouds(df, text_column='comment_text'):
    """
    Generate word clouds for toxic and non-toxic comments
    """
    # Define toxic comments as those with any toxic label
    df['is_toxic'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0).astype(int)
    
    # Get toxic and non-toxic comments
    toxic_comments = ' '.join(df[df['is_toxic'] == 1][text_column].astype(str))
    non_toxic_comments = ' '.join(df[df['is_toxic'] == 0][text_column].astype(str))
    
    # Generate word clouds
    plt.figure(figsize=(16, 8))
    
    # Toxic comments word cloud
    plt.subplot(1, 2, 1)
    wordcloud_toxic = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(toxic_comments)
    plt.imshow(wordcloud_toxic, interpolation='bilinear')
    plt.title('Most Common Words in Toxic Comments', fontsize=16)
    plt.axis('off')
    
    # Non-toxic comments word cloud
    plt.subplot(1, 2, 2)
    wordcloud_non_toxic = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(non_toxic_comments)
    plt.imshow(wordcloud_non_toxic, interpolation='bilinear')
    plt.title('Most Common Words in Non-Toxic Comments', fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('word_clouds.png')
    plt.show()

def plot_correlation_matrix(df, label_columns):
    """
    Plot correlation matrix between different toxic labels
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[label_columns].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Between Different Toxic Labels', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load the dataset
    train_df = pd.read_csv('train.csv/train.csv')
    
    # Define label columns
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Print original label distribution
    print("Original label distribution:")
    for col in label_columns:
        print(f"{col}: {train_df[col].sum()} ({train_df[col].mean()*100:.2f}%)")
    
    # Create visualizations for the original dataset
    print("\nGenerating visualizations for the original dataset...")
    plot_label_distribution(train_df, label_columns)
    plot_comment_length_distribution(train_df)
    generate_wordclouds(train_df)
    plot_correlation_matrix(train_df, label_columns)
    
    # Preprocess the dataset
    processed_df = preprocess_dataset(train_df, augment=True)
    
    # Print new label distribution
    print("\nNew label distribution after augmentation:")
    for col in label_columns:
        print(f"{col}: {processed_df[col].sum()} ({processed_df[col].mean()*100:.2f}%)")
    
    # Create visualizations for the processed dataset
    print("\nGenerating visualizations for the processed dataset...")
    plot_label_distribution(processed_df, label_columns)
    plot_comment_length_distribution(processed_df)
    
    # Save the preprocessed dataset
    processed_df.to_csv('preprocessed_train.csv', index=False)
    print("\nPreprocessed dataset saved to 'preprocessed_train.csv'") 
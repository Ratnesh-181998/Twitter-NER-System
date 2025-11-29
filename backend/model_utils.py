import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from torch.utils.data import Dataset, DataLoader
import logging
import os
from collections import defaultdict

# Setup logging
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class NERDataset(Dataset):
    """Custom Dataset for NER"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NERModel:
    def __init__(self, model_type='bert', model_name='bert-base-uncased', data_dir='../../'):
        self.model_type = model_type
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.schema = None
        self.tag2id = None
        self.id2tag = None
        self.max_len = 128
        self.tokenizer = None
        self.device = device
        
        if model_type == 'lstm':
            # Placeholder for LSTM initialization
            pass
        else:
            # Assume any other type is a Transformer model name
            # If model_type is 'bert', use the default model_name, otherwise use model_type as the name
            if model_type == 'bert':
                self.model_name = 'bert-base-uncased'
            else:
                self.model_name = model_type
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logger.info(f"Initialized {self.model_name} model on {self.device}")

    def load_data(self, filename):
        """Load CoNLL format data"""
        filepath = os.path.join(self.data_dir, filename)
        logger.info(f"Loading data from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = [line.strip().split() for line in file]
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return []
        
        samples, start = [], 0
        for end, parts in enumerate(lines):
            if not parts:
                sample = []
                for item in lines[start:end]:
                    if len(item) == 2:
                        sample.append((item[0], item[1]))
                if sample:
                    samples.append(sample)
                start = end + 1
        
        # Handle last sample
        if start < len(lines):
            sample = []
            for item in lines[start:]:
                if len(item) == 2:
                    sample.append((item[0], item[1]))
            if sample:
                samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {filename}")
        return samples

    def prepare_data(self):
        """Prepare training and test data"""
        train_samples = self.load_data('wnut 16.txt.conll')
        test_samples = self.load_data('wnut 16test.txt.conll')
        
        all_samples = train_samples + test_samples
        
        # Get unique tags
        all_tags = set()
        for sentence in all_samples:
            for _, tag in sentence:
                all_tags.add(tag)
        
        self.schema = sorted(all_tags)
        
        self.tag2id = {tag: i for i, tag in enumerate(self.schema)}
        self.id2tag = {i: tag for i, tag in enumerate(self.schema)}
        
        # Calculate max length
        if all_samples:
            self.max_len = min(max(len(sample) for sample in all_samples), 128)
        else:
            self.max_len = 128
            
        logger.info(f"Max sequence length: {self.max_len}")
        logger.info(f"Number of tags: {len(self.schema)}")
        logger.info(f"Tags: {self.schema}")
        
        return train_samples, test_samples

    def tokenize_and_align_labels(self, samples):
        """Tokenize samples and align labels"""
        tokenized_inputs = []
        labels = []
        
        for sample in samples:
            tokens = [token for token, _ in sample]
            tags = [tag for _, tag in sample]
            
            # Tokenize
            tokenized = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors=None
            )
            
            # Align labels
            word_ids = tokenized.word_ids()
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(self.tag2id[tags[word_idx]])
                else:
                    label_ids.append(-100)  # Sub-tokens
                previous_word_idx = word_idx
            
            tokenized_inputs.append(tokenized)
            labels.append(label_ids)
        
        # Convert to proper format
        encodings = {
            'input_ids': [t['input_ids'] for t in tokenized_inputs],
            'attention_mask': [t['attention_mask'] for t in tokenized_inputs]
        }
        
        return encodings, labels

    def build_bert_model(self):
        """Build BERT model using PyTorch"""
        logger.info("Building PyTorch BERT model...")
        
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.schema),
                id2label=self.id2tag,
                label2id=self.tag2id
            )
            
            self.model.to(self.device)
            logger.info(f"BERT model built successfully with {len(self.schema)} labels")
            logger.info(f"Model moved to {self.device}")
            
        except Exception as e:
            logger.error(f"Error building model: {e}", exc_info=True)
            raise

    def train(self, epochs=3, batch_size=16, learning_rate=5e-5):
        """Train the model"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        train_samples, test_samples = self.prepare_data()
        
        # Prepare data
        train_encodings, train_labels = self.tokenize_and_align_labels(train_samples)
        test_encodings, test_labels = self.tokenize_and_align_labels(test_samples)
        
        # Create datasets
        train_dataset = NERDataset(train_encodings, train_labels)
        test_dataset = NERDataset(test_encodings, test_labels)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Build model if not already built
        if self.model is None:
            self.build_bert_model()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs} - Val Loss: {avg_val_loss:.4f}")
        
        logger.info("Training completed!")
        return True

    def predict(self, sentence):
        """Predict entities in a sentence"""
        if self.model is None:
            raise ValueError("Model not initialized. Please build or load a model first.")
        
        self.model.eval()
        
        words = sentence.split()
        
        # Tokenize
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=self.max_len
        )
        
        # Get word_ids from the first (and only) sequence
        word_ids = tokenized.word_ids(batch_index=0)
        
        # Move tensors to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Align predictions with words
        aligned_labels = []
        current_word = None
        
        for word_id, pred in zip(word_ids, predictions[0].cpu().numpy()):
            if word_id is not None and word_id != current_word:
                current_word = word_id
                tag = self.id2tag.get(pred, 'O')
                aligned_labels.append(tag)
        
        return list(zip(words, aligned_labels))

    def save_model(self, path='saved_model'):
        """Save the trained model"""
        logger.info(f"Saving model to {path}...")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save schema
        import json
        with open(os.path.join(path, 'schema.json'), 'w') as f:
            json.dump({
                'schema': self.schema,
                'tag2id': self.tag2id,
                'id2tag': {str(k): v for k, v in self.id2tag.items()},
                'model_type': self.model_type
            }, f)
        
        logger.info("Model saved successfully")

    def load_saved_model(self, path='saved_model'):
        """Load a saved model"""
        logger.info(f"Loading model from {path}...")
        
        import json
        with open(os.path.join(path, 'schema.json'), 'r') as f:
            data = json.load(f)
            self.schema = data['schema']
            self.tag2id = data['tag2id']
            self.id2tag = {int(k): v for k, v in data['id2tag'].items()}
            self.model_type = data['model_type']
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")

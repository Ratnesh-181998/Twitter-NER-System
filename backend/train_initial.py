"""
Quick script to train and save the BERT model
This bypasses the initialization issues by training the model directly
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from model_utils import NERModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting BERT model training...")
    
    # Initialize model - Data is now in the current directory (backend)
    data_dir = os.path.dirname(__file__)
    model = NERModel(model_type='bert', data_dir=data_dir)
    
    # Train for 3 epochs for better accuracy
    logger.info("Training for 3 epochs...")
    try:
        model.train(epochs=3, batch_size=16)
        
        # Save the model with the correct name the API expects
        save_path = 'bert-base-uncased_ner_model'
        logger.info(f"Saving model to {save_path}...")
        model.save_model(save_path)
        
        logger.info(f"Done! Model saved to {save_path}/")
        logger.info("You can now restart the API and it will load the trained model.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()

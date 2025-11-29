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
    
    # Initialize model
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model = NERModel(model_type='bert', data_dir=data_dir)
    
    # Train for 1 epoch
    logger.info("Training for 1 epoch to initialize the model...")
    model.train(epochs=1, batch_size=32)
    
    # Save the model
    logger.info("Saving model...")
    model.save_model('bert_ner_model')
    
    logger.info("Done! Model saved to bert_ner_model/")
    logger.info("You can now restart the API and it will load the trained model.")

if __name__ == "__main__":
    main()

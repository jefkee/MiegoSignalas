import os
import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from src.models.dataset import Dataset
from src.models.classifier import SleepClassifier
from src.config.settings import MODEL_PARAMS, TRAINING_PARAMS
import gc
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('training')

def main():
    try:
        # Initialize dataset
        data_dir = os.path.join('data', 'raw')
        dataset = Dataset(data_dir)
        
        # Load and preprocess data in smaller batches
        logger.info("Loading dataset...")
        try:
            X, y = dataset.load_training_data(batch_size=3)  # Process 3 files at a time
        except KeyboardInterrupt:
            logger.info("\nData loading interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            sys.exit(1)
        
        if len(X) == 0:
            logger.error("No valid data loaded")
            return
            
        # Free up memory
        gc.collect()
        
        # Print dataset statistics
        logger.info(f"\nDataset loaded successfully:")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        
        # Compute class weights
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Initialize model
        logger.info("\nInitializing model...")
        model = SleepClassifier(MODEL_PARAMS)
        
        try:
            # Train model with memory-efficient batch processing
            logger.info("Starting model training")
            history = model.model.fit(
                X, y,
                batch_size=TRAINING_PARAMS['batch_size'],
                epochs=TRAINING_PARAMS['epochs'],
                validation_split=TRAINING_PARAMS['validation_split'],
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Save model
            logger.info("Saving model...")
            model_dir = os.path.join('models')
            os.makedirs(model_dir, exist_ok=True)
            model.model.save(os.path.join(model_dir, 'sleep_classifier.h5'))
            logger.info("Model saved successfully")
            
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        # Clean up memory
        gc.collect()

if __name__ == '__main__':
    main() 
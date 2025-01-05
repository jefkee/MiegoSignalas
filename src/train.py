import os
import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from src.models.dataset import Dataset
from src.models.classifier import SleepClassifier
from src.config.settings import MODEL_PARAMS, TRAINING_PARAMS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('training')

def main():
    # Initialize dataset
    data_dir = os.path.join('data', 'raw')
    dataset = Dataset(data_dir)
    
    # Load and preprocess data
    logger.info("Loading dataset...")
    X, y = dataset.load_training_data()
    
    if len(X) == 0:
        logger.error("No valid data loaded")
        return
        
    # Compute class weights
    unique_classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    # Initialize and train model
    logger.info("Starting model training")
    model = SleepClassifier(MODEL_PARAMS)
    
    try:
        history = model.train(
            X, y,
            batch_size=TRAINING_PARAMS['batch_size'],
            epochs=TRAINING_PARAMS['epochs'],
            validation_split=TRAINING_PARAMS['validation_split'],
            class_weight=class_weight_dict
        )
        
        # Save model
        model_dir = os.path.join('models')
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, 'sleep_classifier.h5'))
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
        
if __name__ == '__main__':
    main() 
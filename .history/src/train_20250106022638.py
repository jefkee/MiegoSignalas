import os
import numpy as np
from src.models.dataset import SleepDataset
from src.models.trainer import ModelTrainer
from src.utils.logger import Logger
from sklearn.utils.class_weight import compute_class_weight
from src.models.classifier import weighted_categorical_crossentropy

def main():
    # Initialize logger
    logger = Logger('training')
    logger.info('Starting model training')
    
    try:
        # Set paths
        data_dir = os.path.join('data', 'raw')
        
        # Verify data directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}. Please create it and add PSG files.")
            
        # List contents of data directory
        logger.info(f"Contents of {data_dir}:")
        for f in os.listdir(data_dir):
            logger.info(f"  - {f}")
        
        model_dir = os.path.join('models')
        vis_dir = os.path.join(model_dir, 'visualizations')
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        for subdir in ['curves', 'confusion', 'roc', 'models']:
            os.makedirs(os.path.join(vis_dir, subdir), exist_ok=True)
        
        # Load and prepare dataset
        logger.info('Loading dataset...')
        dataset = SleepDataset(data_dir, use_augmentation=True)
        X, y = dataset.load_training_data()
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Dataset is empty after loading")
            
        logger.info(f'Dataset loaded successfully: {len(X)} samples')
        
        # Calculate class weights
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Configure model
        config = {
            'input_shape': [3000, 7],
            'epochs': 150,
            'batch_size': 32,
            'class_weights': class_weight_dict,
            'learning_rate': 0.001,
            'custom_loss': weighted_categorical_crossentropy(class_weight_dict),
            'early_stopping': {
                'patience': 20,
                'min_delta': 0.0005,
                'monitor': 'val_loss',
                'restore_best_weights': True,
                'mode': 'min'
            },
            'reduce_lr': {
                'factor': 0.5,
                'patience': 8,
                'min_lr': 0.00001
            }
        }
        
        # Train with cross-validation
        trainer = ModelTrainer(config)
        fold_scores = trainer.train_with_cross_validation(X, y, n_folds=5)
        
        # Save model
        model_path = os.path.join(model_dir, 'sleep_classifier.h5')
        trainer.save_model(model_path)
        logger.info(f'Model saved to {model_path}')
        
        return fold_scores
        
    except Exception as e:
        logger.error(f'Error during training: {str(e)}')
        raise

if __name__ == '__main__':
    main() 
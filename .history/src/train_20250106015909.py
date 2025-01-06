import os
import numpy as np
from src.models.dataset import SleepDataset
from src.models.trainer import ModelTrainer
from src.utils.logger import Logger
from sklearn.utils.class_weight import compute_class_weight
from src.models.classifier import SleepStageClassifier

def main():
    # Initialize logger
    logger = Logger('training')
    logger.info('Starting model training')
    
    try:
        # Set paths
        data_dir = os.path.join('data', 'raw')
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
        logger.info(f'Dataset loaded: {len(X)} samples')
        
        # Calculate class weights
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Log class distribution and weights
        logger.info('\nDetailed class distribution:')
        for cls in unique_classes:
            count = np.sum(y == cls)
            logger.info(f'Class {cls}: {count} samples ({count/len(y)*100:.2f}%)')
        
        logger.info('\nClass weights after smoothing:')
        for cls, weight in class_weight_dict.items():
            logger.info(f'Class {cls}: {weight:.4f}')
        
        # Configure model
        config = {
            'n_channels': 7,
            'epochs': 150,
            'batch_size': 32,
            'class_weights': list(class_weights),  # Convert to list for the model
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
import os
import numpy as np
from src.models.dataset import SleepDataset
from src.models.trainer import ModelTrainer
from src.utils.logger import Logger
from sklearn.utils.class_weight import compute_class_weight
from src.models.classifier import SleepStageClassifier
import tensorflow as tf

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
        
        # Normalize data to [-1, 1]
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        # Calculate class weights
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        logger.info("\nClass weights:")
        for i, weight in class_weight_dict.items():
            logger.info(f"Class {i}: {weight:.4f}")
        
        # Configure training
        config = {
            'input_shape': [3000, 7],  # 30 seconds at 100Hz, 7 channels
            'batch_size': 32,
            'epochs': 50,
            'class_weights': class_weight_dict,
            'validation_split': 0.2,
            'shuffle': True,
            'callbacks': [
                # Early stopping
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    mode='min'
                ),
                # Learning rate scheduler
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                ),
                # Model checkpoint
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(model_dir, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                # TensorBoard logging
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(vis_dir, 'tensorboard'),
                    histogram_freq=1
                )
            ]
        }
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Train with cross-validation
        logger.info("Starting cross-validation training...")
        fold_scores = trainer.train_with_cross_validation(X, y, n_folds=5)
        
        # Log results
        logger.info("\nCross-validation results:")
        for fold, score in enumerate(fold_scores, 1):
            logger.info(f"Fold {fold}: {score:.4f}")
        logger.info(f"Mean accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        # Save final model
        model_path = os.path.join(model_dir, 'sleep_classifier.h5')
        trainer.save_model(model_path)
        logger.info(f'Model saved to {model_path}')
        
        return fold_scores
        
    except Exception as e:
        logger.error(f'Error during training: {str(e)}')
        raise

if __name__ == '__main__':
    main() 
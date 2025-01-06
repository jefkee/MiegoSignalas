import tensorflow as tf
from sklearn.model_selection import KFold
from .classifier import SleepStageClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import os

class ModelTrainer:
    """Model trainer with cross-validation support"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def train_with_cross_validation(self, X, y, n_folds=5):
        """Train model with k-fold cross validation"""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\nTraining Fold {fold}/{n_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Initialize model for this fold
            self.model = SleepStageClassifier(
                input_shape=self.config['input_shape']
            )
            
            # Train model
            history = self.model.model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=self.config['callbacks'],
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_acc = self.model.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(val_acc)
            
            # Generate predictions for visualization
            y_pred = self.model.model.predict(X_val)
            
            # Save visualizations
            self._save_fold_visualizations(fold, history, y_val, y_pred)
            
            # Save fold model
            self.model.model.save(os.path.join('models', f'model_fold_{fold}.h5'))
            
        return fold_scores
    
    def save_model(self, path):
        """Save the best model"""
        if self.model:
            self.model.model.save(path)
    
    def _save_fold_visualizations(self, fold, history, y_true, y_pred):
        """Save training visualizations for each fold"""
        vis_dir = os.path.join('models', 'visualizations')
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'Model Accuracy - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Model Loss - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'curves', f'training_history_fold_{fold}.png'))
        plt.close()
        
        # Confusion matrix
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
                   yticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'])
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(vis_dir, 'confusion', f'confusion_matrix_fold_{fold}.png'))
        plt.close()
        
        # ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(5):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - Fold {fold}')
        plt.legend()
        plt.savefig(os.path.join(vis_dir, 'roc', f'roc_curves_fold_{fold}.png'))
        plt.close()
        
        # Save classification report
        report = classification_report(y_true, y_pred_classes)
        with open(os.path.join(vis_dir, f'classification_report_fold_{fold}.txt'), 'w') as f:
            f.write(report)
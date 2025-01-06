import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from .classifier import SleepStageClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from itertools import cycle
import os

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def train_with_cross_validation(self, X, y, n_folds=5):
        """Train model with cross-validation"""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nTraining fold {fold + 1}/{n_folds}")
            
            # Get fold data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Initialize model for this fold
            self.model = SleepStageClassifier(
                n_channels=self.config['n_channels'],
                class_weights=self.config['class_weights']
            )
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    **self.config['early_stopping']
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    **self.config['reduce_lr']
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('models', 'visualizations', 'models', f'best_model_fold{fold+1}.h5'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train the model
            history = self.model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate fold
            scores = self.model.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(scores)
            
            # Plot training curves
            self._plot_learning_curves(history, fold)
            
            # Plot confusion matrix and ROC curves
            y_pred = self.model.predict(X_val)
            self._plot_confusion_matrix(y_val, y_pred.argmax(axis=1), fold)
            self._plot_roc_curves(y_val, y_pred, fold)
        
        return fold_scores
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is not None:
            self.model.model.save(path)
    
    def _plot_learning_curves(self, history, fold):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy (Fold {fold+1})')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss (Fold {fold+1})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.savefig(os.path.join('models', 'visualizations', 'curves', f'learning_curves_fold{fold+1}.png'))
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred, fold):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Fold {fold+1})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join('models', 'visualizations', 'confusion', f'confusion_matrix_fold{fold+1}.png'))
        plt.close()
    
    def _plot_roc_curves(self, y_true, y_pred_proba, fold):
        """Plot ROC curves for each class"""
        n_classes = y_pred_proba.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Calculate ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (Fold {fold+1})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('models', 'visualizations', 'roc', f'roc_curves_fold{fold+1}.png'))
        plt.close()
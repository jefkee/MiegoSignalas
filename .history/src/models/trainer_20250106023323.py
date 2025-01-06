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
        """Train model using cross-validation"""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        print("\nStarting cross-validation training...")
        print(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nTraining fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Initialize new model for each fold
            self.model = SleepStageClassifier(input_shape=self.config['input_shape'])
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['early_stopping']['patience'],
                    min_delta=self.config['early_stopping']['min_delta'],
                    restore_best_weights=True,
                    mode='min'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=self.config['reduce_lr']['factor'],
                    patience=self.config['reduce_lr']['patience'],
                    min_lr=self.config['reduce_lr']['min_lr']
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join('models', 'visualizations', 'models', f'best_model_fold{fold+1}.h5'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train the model
            history = self.model.model.fit(
                X_train, 
                y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=self.config['class_weights'],
                verbose=1
            )
            
            # Evaluate fold
            val_loss, val_acc = self.model.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(val_acc)
            
            print(f"Fold {fold + 1} - Validation Accuracy: {val_acc:.4f}")
            
            # Save learning curves
            self._plot_learning_curves(history, fold)
            
            # Save confusion matrix and ROC curves
            y_pred = self.model.model.predict(X_val)
            self._plot_confusion_matrix(y_val, np.argmax(y_pred, axis=1), fold)
            self._plot_roc_curves(y_val, y_pred, fold)
        
        print("\nCross-validation complete")
        print(f"Mean validation accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        return fold_scores
    
    def save_model(self, path):
        self.model.model.save(path)
    
    def _get_callbacks(self, fold):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join('models', 'visualizations', 'models', f'best_model_fold{fold}.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join('models', 'visualizations', 'curves', f'training_log_fold{fold}.csv')
            )
        ]
    
    def _plot_learning_curves(self, history, fold):
        """Plot training and validation learning curves"""
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
        """Plot ROC curves"""
        n_classes = y_pred_proba.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
        
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
    
    def _save_results_summary(self, fold_scores, confusion_matrices, reports):
        """Save a summary of all results"""
        with open(os.path.join('models', 'visualizations', 'training_results.txt'), 'w') as f:
            f.write("Sleep Stage Classification Results\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Overall Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}\n\n")
            
            # Average confusion matrix
            avg_cm = np.mean(confusion_matrices, axis=0)
            f.write("Average Confusion Matrix:\n")
            f.write(str(avg_cm) + "\n\n")
            
            # Per-class metrics
            f.write("Per-class Performance:\n")
            f.write(reports[-1])  # Write the last fold's detailed report
    
    def load_ensemble(self, n_folds=5):
        """Load all fold models for ensembling"""
        models = []
        for fold in range(n_folds):
            model_path = os.path.join('models', 'visualizations', 'models', f'best_model_fold{fold+1}.h5')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                models.append(model)
        return models
    
    def predict_ensemble(self, X, models):
        """Make predictions using model ensemble"""
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
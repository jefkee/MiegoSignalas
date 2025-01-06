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
        """Apmoko modelį naudojant kryžminę validaciją"""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            # Apmokymas kiekvienam fold'ui
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model = SleepStageClassifier(input_shape=self.config['input_shape'])
            history = self.model.model.fit(...)
            
            # Saugo rezultatus ir vizualizacijas
            self._plot_learning_curves(history, fold)
            self._evaluate_fold(self.model, X_val, y_val, fold)
    
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
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy (Fold {fold})')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss (Fold {fold})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join('models', 'visualizations', 'curves', f'learning_curves_fold{fold}.png'))
        plt.close()
    
    def _evaluate_fold(self, model, X_val, y_val, fold):
        """Evaluate model performance for a fold"""
        # Get predictions
        y_pred = model.predict(X_val).argmax(axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Fold {fold})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join('models', 'visualizations', 'confusion', f'confusion_matrix_fold{fold}.png'))
        plt.close()
        
        # Print classification report
        report = classification_report(y_val, y_pred, target_names=[
            'Wake', 'N1', 'N2', 'N3', 'REM'
        ])
        print(f"\nClassification Report (Fold {fold}):")
        print(report)
        
        return cm, report
    
    def _plot_roc_curves(self, y_true, y_pred_proba, fold):
        """Plot ROC curves for each class"""
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
        plt.title(f'ROC Curves (Fold {fold})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('models', 'visualizations', 'roc', f'roc_curves_fold{fold}.png'))
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
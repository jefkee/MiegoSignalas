import tensorflow as tf
from sklearn.model_selection import train_test_split
from .classifier import SleepStageClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def train(self, X, y):
        # Create and compile model
        self.model = SleepStageClassifier(input_shape=self.config['input_shape'])
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        history = self.model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            class_weight=self.config.get('class_weights'),
            callbacks=self._get_callbacks(),
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_acc = self.model.model.evaluate(X_val, y_val)
        print(f"\nValidation accuracy: {val_acc:.4f}")
        
        # Plot learning curves
        self._plot_learning_curves(history)
        
        # Plot confusion matrix
        y_pred = np.argmax(self.model.model.predict(X_val), axis=1)
        self._plot_confusion_matrix(y_val, y_pred)
        
        return val_acc
        
    def _get_callbacks(self):
        callbacks = []
        
        # Early stopping
        if 'early_stopping' in self.config:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                **self.config['early_stopping']
            )
            callbacks.append(early_stopping)
            
        return callbacks
        
    def _plot_learning_curves(self, history):
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        os.makedirs('models/visualizations', exist_ok=True)
        plt.savefig('models/visualizations/learning_curves.png')
        plt.close()
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        
        classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save confusion matrix plot
        plt.savefig('models/visualizations/confusion_matrix.png')
        plt.close()
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=classes)
        with open('models/visualizations/training_results.txt', 'w') as f:
            f.write('Sleep Stage Classification Results\n')
            f.write('='*30 + '\n\n')
            f.write(report)
            
    def save_model(self, path):
        """Save the trained model to disk"""
        if self.model and self.model.model:
            self.model.model.save(path)
        else:
            raise ValueError("No model to save")
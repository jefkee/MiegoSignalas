import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, LSTM,
                         BatchNormalization, Bidirectional, GlobalAveragePooling1D,
                         Flatten)
class SleepStageClassifier:
    """
    Neural network for sleep stage classification
    """
    def __init__(self, input_shape=None):
        self.input_shape = input_shape or [3000, 7]  # Default: 30s @ 100Hz, 7 channels
        self.model = self._build_model(self.input_shape)
        
    def _build_model(self, input_shape):
        # Input layer
        inputs = tf.keras.Input(shape=input_shape)
        x = BatchNormalization()(inputs)
        
        # CNN blokas 1 - mažesni filtrai, daugiau sluoksnių
        x = Conv1D(32, kernel_size=32, strides=2, activation='relu', padding='same')(x)
        x = Conv1D(32, kernel_size=32, strides=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        
        # CNN blokas 2
        x = Conv1D(64, kernel_size=16, strides=2, activation='relu', padding='same')(x)
        x = Conv1D(64, kernel_size=16, strides=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        
        # CNN blokas 3
        x = Conv1D(128, kernel_size=8, strides=1, activation='relu', padding='same')(x)
        x = Conv1D(128, kernel_size=8, strides=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        
        # LSTM blokas
        x = Bidirectional(LSTM(64, return_sequences=True,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Global Average Pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense blokai - mažiau neuronų
        x = Dense(128, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        outputs = Dense(5, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.0002,  # Mažesnis learning rate
            weight_decay=0.05,     # Didesnis weight decay
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model

    @classmethod
    def load_model(cls, model_path):
        """Load a trained model from file"""
        instance = cls()
        instance.model = tf.keras.models.load_model(model_path)
        return instance

    def predict(self, features):
        """Make predictions on preprocessed features"""
        try:
            print(f"Input features shape: {features.shape}")
            
            # Model expects (batch_size, timesteps, features)
            if len(features.shape) != 3:
                raise ValueError(f"Expected 3D input array (batch, time, channels), got shape {features.shape}")
            
            predictions = self.model.predict(features)
            print(f"Predictions shape: {predictions.shape}")
            
            return predictions.squeeze()
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            raise        
def attention_block(x):
    # Self attention su reguliarizacija
    attention = tf.keras.layers.Dense(
        x.shape[-1], 
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    attention = tf.keras.layers.LeakyReLU(alpha=0.3)(attention)
    attention = tf.keras.layers.Dense(
        1, 
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(attention)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention)
    
    # Apply attention
    context_vector = tf.keras.layers.Multiply()([x, attention_weights])
    return context_vector

def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights = tf.gather(class_weights, y_true)
        return tf.reduce_mean(
            weights * tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        )
    return loss


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, LSTM,
                         BatchNormalization, Bidirectional, GlobalAveragePooling1D)
class SleepStageClassifier:
    """
    Neural network for sleep stage classification
    """
    def __init__(self, input_shape=None):
        self.input_shape = input_shape or [3000, 7]  # Default shape: 30s @ 100Hz, 7 channels
        self.model = self._build_model(self.input_shape, custom_loss=None)
        
    def _build_model(self, input_shape, custom_loss=None):
        # Input layer
        inputs = tf.keras.Input(shape=input_shape)
        x = BatchNormalization()(inputs)
        
        # First conv block
        conv1 = Conv1D(32, kernel_size=64, strides=1, padding='same', activation='relu')(x)
        bn1 = BatchNormalization()(conv1)
        conv1_2 = Conv1D(32, kernel_size=64, padding='same', activation='relu')(bn1)
        bn1_2 = BatchNormalization()(conv1_2)
        pool1 = MaxPooling1D(pool_size=4)(bn1_2)
        drop1 = Dropout(0.3)(pool1)
        
        # Second block
        conv2 = Conv1D(64, kernel_size=32, strides=1, padding='same', activation='relu')(drop1)
        bn2 = BatchNormalization()(conv2)
        conv2_2 = Conv1D(64, kernel_size=32, padding='same', activation='relu')(bn2)
        bn2_2 = BatchNormalization()(conv2_2)
        pool2 = MaxPooling1D(pool_size=4)(bn2_2)
        drop2 = Dropout(0.3)(pool2)
        
        # Third block
        conv3 = Conv1D(128, kernel_size=16, strides=1, padding='same', activation='relu')(drop2)
        bn3 = BatchNormalization()(conv3)
        conv3_2 = Conv1D(128, kernel_size=16, padding='same', activation='relu')(bn3)
        bn3_2 = BatchNormalization()(conv3_2)
        pool3 = MaxPooling1D(pool_size=4)(bn3_2)
        drop3 = Dropout(0.3)(pool3)
        
        # LSTM layers
        lstm1 = Bidirectional(LSTM(64, return_sequences=True))(drop3)
        drop4 = Dropout(0.3)(lstm1)
        
        # Attention
        att = attention_block(drop4)
        
        # Final layers
        lstm2 = Bidirectional(LSTM(64))(att)
        drop5 = Dropout(0.3)(lstm2)
        
        dense1 = Dense(128, activation='relu')(drop5)
        drop6 = Dropout(0.3)(dense1)
        outputs = Dense(5, activation='softmax')(drop6)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use slightly higher learning rate with decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=1000,
            decay_rate=0.95
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss=custom_loss or 'sparse_categorical_crossentropy',
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
    # Self attention
    attention = tf.keras.layers.Dense(x.shape[-1], use_bias=False)(x)
    attention = tf.keras.layers.Activation('tanh')(attention)
    attention = tf.keras.layers.Dense(1, use_bias=False)(attention)
    attention = tf.keras.layers.Softmax(axis=1)(attention)
    
    # Apply attention
    return tf.keras.layers.Multiply()([x, attention])

def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights = tf.gather(class_weights, y_true)
        return tf.reduce_mean(
            weights * tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        )
    return loss
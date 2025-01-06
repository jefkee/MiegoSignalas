import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, LSTM,
                         BatchNormalization, Bidirectional, GlobalAveragePooling1D,
                         Flatten, LayerNormalization)
class SleepStageClassifier:
    """
    Neural network for sleep stage classification with variable-length input support
    """
    def __init__(self, n_channels=7):
        self.n_channels = n_channels
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layer - None allows variable length
        inputs = tf.keras.Input(shape=(None, self.n_channels))
        x = BatchNormalization()(inputs)
        
        # CNN blocks for feature extraction
        # 1D convolutions can handle variable length
        x = self._conv_block(x, filters=64, kernel_size=32, dilation_rate=1)
        x = self._conv_block(x, filters=128, kernel_size=16, dilation_rate=2)
        x = self._conv_block(x, filters=128, kernel_size=8, dilation_rate=4)
        
        # Bidirectional LSTM for temporal dependencies
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Multi-head self attention
        x = self._multi_head_attention(x, num_heads=8)
        
        # Global pooling to handle variable length
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers for classification
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        outputs = Dense(5, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with AdamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _conv_block(self, x, filters, kernel_size, dilation_rate):
        """Convolutional block with residual connection"""
        # Shortcut connection
        shortcut = x
        
        # First conv layer
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)
        
        # Second conv layer
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = BatchNormalization()(x)
        
        # Project shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add residual connection
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        return x
    
    def _multi_head_attention(self, x, num_heads):
        """Multi-head self attention block"""
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=x.shape[-1] // num_heads,
            dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = Dense(x.shape[-1] * 2, activation='relu')(x)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(x.shape[-1])(ffn)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        return x
    
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
            predictions = self.model.predict(features)
            print(f"Predictions shape: {predictions.shape}")
            return predictions.squeeze()
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            raise


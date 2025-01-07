import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, LSTM,
                         BatchNormalization, Bidirectional, GlobalAveragePooling1D,
                         Flatten, Activation)

class ResizeSequence(tf.keras.layers.Layer):
    def __init__(self, target_len, **kwargs):
        super().__init__(**kwargs)
        self.target_len = target_len

    def call(self, inputs):
        return tf.image.resize(
            tf.expand_dims(inputs, 2),
            (self.target_len, 1),
            method='bilinear'
        )[:, :, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_len, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({"target_len": self.target_len})
        return config

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
        print("After input:", x.shape)
        
        # First CNN block
        x1 = Conv1D(64, kernel_size=64, strides=2, activation='relu', padding='same')(x)
        print("After first conv:", x1.shape)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)
        print("After first pool (x1):", x1.shape)
        x1_skip = Dropout(0.3)(x1)
        print("After dropout (x1_skip):", x1_skip.shape)
        
        # Second CNN block
        x2 = Conv1D(128, kernel_size=32, strides=2, activation='relu', padding='same')(x1_skip)
        print("After second conv:", x2.shape)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling1D(pool_size=2)(x2)
        print("After second pool:", x2.shape)
        x2 = Dropout(0.3)(x2)
        
        # Third CNN block
        x3 = Conv1D(256, kernel_size=16, strides=2, activation='relu', padding='same')(x2)
        print("After third conv:", x3.shape)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling1D(pool_size=2)(x3)
        print("After third pool (x3 final):", x3.shape)
        
        # Skip connection with exact size matching
        x_skip = Conv1D(256, kernel_size=1)(x1_skip)
        print("After skip conv:", x_skip.shape)
        
        # Use custom layer to resize x_skip to match x3's length
        x_skip = ResizeSequence(47)(x_skip)
        print("After skip resize (x_skip final):", x_skip.shape)
        
        # Add skip connection
        x = tf.keras.layers.Add()([x3, x_skip])
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        # Bidirectional LSTM with increased units
        x = Bidirectional(LSTM(128, 
                             return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.l2(0.001),
                             recurrent_regularizer=tf.keras.regularizers.l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        attention = Dense(256, activation='tanh')(x)
        attention = tf.keras.layers.Softmax(axis=1)(attention)
        x = tf.keras.layers.Multiply()([x, attention])
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers with increased capacity
        x = Dense(256, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(5, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Print model summary to verify shapes
        model.summary()
        
        # Optimizer with learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Custom loss function
        def loss_fn(y_true, y_pred):
            # Ensure y_true is the right shape and type
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.reshape(y_true, [-1])  # Flatten to 1D
            y_true_1_hot = tf.one_hot(y_true, depth=5)
            
            # Label smoothing
            smooth_factor = 0.1
            y_true_1_hot = y_true_1_hot * (1.0 - smooth_factor) + (smooth_factor / 5.0)
            
            # Calculate cross entropy
            loss = tf.keras.losses.categorical_crossentropy(
                y_true_1_hot,
                y_pred,
                from_logits=False
            )
            
            return tf.reduce_mean(loss)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        return model

    @classmethod
    def load_model(cls, model_path):
        """Load a trained model"""
        instance = cls()
        
        # Define the loss function
        def loss_fn(y_true, y_pred):
            # Ensure y_true is the right shape and type
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.reshape(y_true, [-1])  # Flatten to 1D
            y_true_1_hot = tf.one_hot(y_true, depth=5)
            
            # Label smoothing
            smooth_factor = 0.1
            y_true_1_hot = y_true_1_hot * (1.0 - smooth_factor) + (smooth_factor / 5.0)
            
            # Calculate cross entropy
            loss = tf.keras.losses.categorical_crossentropy(
                y_true_1_hot,
                y_pred,
                from_logits=False
            )
            
            return tf.reduce_mean(loss)
        
        # Create custom objects dictionary with both loss function and ResizeSequence layer
        custom_objects = {
            'loss_fn': loss_fn,
            'ResizeSequence': ResizeSequence
        }
        
        # Load the model with custom objects
        try:
            with tf.keras.utils.custom_object_scope(custom_objects):
                instance.model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Initialize a new model if loading fails
            instance.model = instance._build_model(instance.input_shape)
            print("Initialized new model instead")
            
        return instance

    def predict(self, features):
        """Make predictions on preprocessed features"""
        try:
            print(f"Input features shape: {features.shape}")
            
            # Model expects (batch_size, timesteps, features)
            if len(features.shape) != 3:
                raise ValueError(f"Expected 3D input array (batch, time, channels), got shape {features.shape}")
            
            # Use larger batch size and enable parallel processing
            predictions = self.model.predict(
                features,
                batch_size=128,  # Increased from default
                verbose=1,
                workers=4,       # Enable parallel processing
                use_multiprocessing=True
            )
            print(f"Predictions shape: {predictions.shape}")
            
            return predictions
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            raise

def attention_block(x):
    """Multi-head self attention with improved scaling"""
    # Split input into multiple heads
    num_heads = 4
    head_dim = x.shape[-1] // num_heads
    
    heads = []
    for i in range(num_heads):
        # Project input to query, key, value spaces
        query = tf.keras.layers.Dense(
            head_dim, 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        key = tf.keras.layers.Dense(
            head_dim, 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        value = tf.keras.layers.Dense(
            head_dim, 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(head_dim, tf.float32))
        attention_weights = tf.keras.layers.Softmax(axis=-1)(attention_scores)
        attention_weights = tf.keras.layers.Dropout(0.2)(attention_weights)
        
        # Apply attention to values
        head_output = tf.matmul(attention_weights, value)
        heads.append(head_output)
    
    # Concatenate heads
    multi_head = tf.keras.layers.Concatenate()(heads)
    
    # Project back to original dimension
    output = tf.keras.layers.Dense(
        x.shape[-1],
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(multi_head)
    
    # Residual connection with layer norm
    output = tf.keras.layers.Add()([x, output])
    output = tf.keras.layers.LayerNormalization()(output)
    
    return output

def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights = tf.gather(class_weights, y_true)
        return tf.reduce_mean(
            weights * tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        )
    return loss
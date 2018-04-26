import keras

def build_fn(input_size=10, dense_layers=(10,10), activation='linear', use_linear_block=True):
    """Creates a Keras NN model
    
    Args:
        input_size (int) - Number of features in input vector
        dense_layers ([int]) - Number of units in the dense layers
        activation (str) - Activation function in the dense layers
        use_linear_block (bool) - Whether to use the linear regression block
    """
    # Input layer
    inputs = keras.layers.Input(shape=(input_size,), name='input')
    
    # Make the dense layer
    dense_layer = inputs
    for layer_size in dense_layers:
        dense_layer = keras.layers.Dense(layer_size, activation=activation)(dense_layer)
    
    if use_linear_block:
        # Add the LR layer
        combined_layer = keras.layers.concatenate([dense_layer, inputs])
    else:
        combined_layer = dense_layer
    
    # Output layer
    outputs = keras.layers.Dense(1, activation='linear', name='output')(combined_layer)
    
    # Make/compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_absolute_error',
                  optimizer='rmsprop',
                 metrics=['mean_absolute_error'])
    return model
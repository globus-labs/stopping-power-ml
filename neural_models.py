from keras.layers import Dense, SimpleRNN, RepeatVector, Dropout
from keras.layers import TimeDistributed, LSTM, GRU, Activation, BatchNormalization
from keras.models import Sequential

import numpy as np

def create_sample_dataset(t, n_dim=2):
    sin = np.sin(2 * np.pi * t)
    cos = np.cos(2 * np.pi * t)

    #Stack data into feature columns
    combined = np.column_stack([sin, cos])

    #Reshape the data to be
    return combined.reshape((combined.shape[0],n_dim))


def create_lstm_model(steps_before, steps_after, feature_count, hidden_neurons=300):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    model.add(RepeatVector(steps_after))
    model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_gru_model(steps_before, steps_after, feature_count, hidden_neurons=300):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    model = Sequential()
    model.add(GRU(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    model.add(RepeatVector(steps_after))
    model.add(GRU(output_dim=hidden_neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    return model

def get_mlp_base(X, activation="linear"):
    model = Sequential()
    model.add(Dense(512, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    return model


def get_mlp_wide(X, activation="tanh"):
    model = Sequential()
    model.add(Dense(4096, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(1024, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    return model


def get_mlp_small(X, activation="tanh"):
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    print(model.summary())
    return model

def get_mlp_vsmall(X, activation="tanh"):
    model = Sequential()
    model.add(Dense(2, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    print(model.summary())
    return model

def get_mlp_logan(X, activation="linear"):
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adagrad', metrics=['mae', 'mse'])
    return model

def get_mlp_deep(X, activation="tanh"):
    model = Sequential()
    model.add(Dense(1024, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss="mae", optimizer='sgd', metrics=['mae'])
    return model

def create_schleife_model(steps_before, steps_after, hidden_neurons, feature_count):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    out_neurons = 1

    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    model.add(RepeatVector(steps_after))
    #model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    model.add(LSTM(output_dim=hidden_neurons, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

def create_schleife_single_lstm_model(steps_before, steps_after,
                                        hidden_neurons, feature_count,
                                        activation="tanh"):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    out_neurons = 1

    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False, activation=activation))
    model.add(RepeatVector(steps_after))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

def create_schleife_simple_rnn_model2(hidden_neurons, feature_count, activation="relu", steps_after=1):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    out_neurons = 1

    model = Sequential()
    model.add(SimpleRNN(units=hidden_neurons, input_dim=feature_count, return_sequences=False, activation=activation))
    model.add(RepeatVector(steps_after))
    model.add(SimpleRNN(units=hidden_neurons, input_dim=feature_count, return_sequences=False, activation=activation))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
    print(model.summary())
    return model

def create_schleife_simple_rnn_model(hidden_neurons, feature_count, activation="relu"):
    """
        creates, compiles and returns a RNN model
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """

    model = Sequential()
    model.add(SimpleRNN(units=hidden_neurons, input_dim=feature_count,
                        return_sequences=False, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    print(model.summary())
    return model

def keras_schleife_reshape(data_x,data_y, n_pre, n_post):
    dX, dY = [], []
    for i in range(len(data_x)-n_pre-n_post):
        dX.append(data_x[i:i+n_pre])
        dY.append(data_y[i+n_pre:i+n_pre+n_post])

    dataX = np.array(dX)
    dataY = np.array(dY)
    return dataX, dataY

def train_schleife_model(model, dataX, dataY, epoch_count,validation_split=0.1, verbose=False):
    """
        trains only the sinus model
    """
    history = model.fit(dataX, dataY, batch_size=200, epochs=epoch_count, validation_split=validation_split, verbose=verbose)
    return history

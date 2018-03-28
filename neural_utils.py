import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

def get_data_columns(data):
    charge_attrs = [x for x in data.columns if x.startswith('density')]
    log_charge_attrs = [x for x in data.columns if x.startswith('log')]
    agni_attrs = [x for x in data.columns if x.startswith('AGNI')]
    engineered_attrs = [x for x in data.columns  if x.startswith('eng_')]
    ewald_attrs = ['ion-ion repulsion',]
    velocity_attrs = [x for x in data.columns if x == 'velocity_mag']

    y_col = 'force'
    X_cols = engineered_attrs + charge_attrs + log_charge_attrs + agni_attrs  + ewald_attrs + velocity_attrs
    return (data[X_cols].values, data[y_col].values)


def time_split(X, y, split):
    X_train = X[0:split]
    X_test = X[split:]

    y_train = y[0:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test


def composite_predict(models, X):
    preds = []
    n_models = 1
    if type(models) is not list:
        models = [models]

    for i in range(0, len(models)):
         preds.append(models[i].predict(X))
    composite_pred = np.mean(preds, axis=0)
    return composite_pred


def predict_summarize(models, X, y):
    preds = composite_predict(models, X)
    mae = mean_absolute_error(preds,y)
    stopping_power = np.mean(preds)
    return(preds, mae, stopping_power)


def keras_rnn_reshape(data, n_pre, n_post):
    dX, dY = [], []
    for i in range(len(data)-n_pre-n_post):
        dX.append(data[i:i+n_pre])
        dY.append(data[i+n_pre:i+n_pre+n_post])
    dataX = np.array(dX)
    dataY = np.array(dY)
    return dataX, dataY


def summary_plots(pred_y, y, end=0.9):
    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(15, 10)

    # Full plot
    ax1.plot(range(0, len(pred_y)), pred_y, linewidth=8, alpha=0.35, label="Model")
    ax1.plot(range(0, len(pred_y)), y, linestyle='--', linewidth=3, label="Data")
    ax1.legend()

    # Zoomed plot
    ax2.plot(range(0, len(pred_y)), pred_y, linewidth=8, alpha=0.5)
    ax2.plot(range(0, len(pred_y)), y, linestyle='--', linewidth=3)
    ax2.set_xlim(int(float(len(pred_y)) - 0.1*len(pred_y)),0.9*len(pred_y))
    #ax2.set_ylim(-.5,0.9)
    sns.despine()

def train_model(model, dataX, dataY, epoch_count):
    """
        trains only the sinus model
    """
    print("train model")
    history = model.fit(dataX, dataY, batch_size=10, epochs=epoch_count, validation_split=0.10)
    return history


def train_over_times(model, X, y, splits=[], epochs=10, batch_size=50, verbose=1):
    results = []
    for i in splits:
        X_train, X_test, y_train, y_test = time_split(X, y, i)
        h = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        preds, mae, sp = predict_summarize(model, X, y)
        results.append({"model":model, "pred_y":preds,
                        "split":i, "mae":mae, "batch_size":batch_size,
                        "epochs":epochs, "history":h.history,
                        "stopping_power":sp})
    return (model, results)

def transfer_train(model, X, y, epochs=25, splits=[], batch_size=100, verbose=True):
    model, r = train_over_times(model, X, y, splits=splits, batch_size=batch_size,
                                epochs=epochs, verbose=verbose)
    return (model, pd.DataFrame(r))

def expand_class(X, y, length):
    if len(X) >= length:
        return X, y

    samples = length - len(X)
    X_exp = np.vstack([X, X[0:samples]])
    y_exp = np.concatenate([y, y[0:samples]])

    if len(X_exp) <= length:
        X_exp, y_exp = expand_class(X_exp,y_exp, length)

    return X_exp, y_exp

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random




def check_model():
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 20000)
    ])
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    return model,lr_schedule



def get_model(lr):
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    # batch_size = 16


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 9]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32,activation= "relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 20000)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer= "adam",
                  metrics=["mae"])
    return model


#
# def model_forecast(model, val_data):
#     forecast = []
#     for x in val_data:
#         for x_t in x:
#             x_t = np.reshape(x_t,(1,1,1))
#             prediction = model.predict(x_t)
#         forecast.append(prediction)
#     return forecast



def prepare_window_data(data,window_size = 10):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    for i in range(0,20):
        for j in range( len(data[0])-window_size):
            window = data[i,j:j+window_size]
            y = data[i,j+window_size]
            train_data.append(window)
            train_label.append(y)
    X_train = np.array(train_data)
    X_train = np.reshape(X_train, (-1, window_size, 1))
    for i in range(20,30):
        temp_data = []
        y_temp = []
        for j in range( len(data[0])-window_size):
            window = data[i,j:j+window_size]
            y = data[i,j+window_size]
            y_temp.append(y)
            temp_data.append(window)
        val_data.append(temp_data)
        val_label.append(y_temp)
    X_val = np.array(val_data)
    X_val = np.reshape(X_val, (10,-1, window_size, 1))
    return X_train,np.array(train_label),X_val,np.array(val_label)



def prepare_window_train_val(data_train,data_val,window_size = 10):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    for i in range(len(data_train)):
        for j in range( len(data_train[0])-window_size):
            window = data_train[i,j:j+window_size]
            y = data_train[i,j+window_size]
            train_data.append(window)
            train_label.append(y)
    X_train = np.array(train_data)
    X_train = np.reshape(X_train, (-1, window_size, 1))
    for i in range(len(data_val)):
        temp_data = []
        y_temp = []
        for j in range( len(data_val[0])-window_size):
            window = data_val[i,j:j+window_size]
            y = data_val[i,j+window_size]
            y_temp.append(y)
            temp_data.append(window)
        val_data.append(temp_data)
        val_label.append(y_temp)
    X_val = np.array(val_data)
    X_val = np.reshape(X_val, (len(data_val),-1, window_size, 1))
    return X_train,np.array(train_label),X_val,np.array(val_label)





def prepare_window_train_val_multivariate(data_train,data_val,window_size = 10):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    for i in range(len(data_train)):
        for j in range( len(data_train[0])-window_size):
            window = data_train[i,j:j+window_size]
            y = data_train[i,j+window_size,0]
            train_data.append(window)
            train_label.append(y)
    X_train = np.array(train_data)
    X_train = np.reshape(X_train, (-1, window_size,data_train.shape[2]))
    for i in range(len(data_val)):
        temp_data = []
        y_temp = []
        for j in range( len(data_val[0])-window_size):
            window = data_val[i,j:j+window_size]
            y = data_val[i,j+window_size,0]
            y_temp.append(y)
            temp_data.append(window)
        val_data.append(temp_data)
        val_label.append(y_temp)
    X_val = np.array(val_data)
    X_val = np.reshape(X_val, (len(data_val),-1, window_size,data_train.shape[2] ))
    return X_train,np.array(train_label),X_val,np.array(val_label)





def prepare_window(data,winow_size,batch_size,shuffle_buffer):
    return windowed_dataset(np.concatenate(data[1:5]),winow_size,batch_size,shuffle_buffer)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    print(series.shape)
    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)



def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(10).prefetch(1)
    forecast = model.predict(ds)
    return forecast





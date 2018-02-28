#!/usr/bin/env python3

import ccxt
from configparser import ConfigParser
import json
import os
import pickle
import redis
import socket
import time
import tempfile
import zlib
from random import shuffle
from requests_futures.sessions import FuturesSession

import numpy as np
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler

import keras.models
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

TIME_FRAMES = ['15m', '1h', '4h', '1d']
FEATURE_SIZE = 5
STEP_SIZE = 5
LSTM_SIZE = FEATURE_SIZE * STEP_SIZE
LSTM_MODEL = 'lstm'
BOT_NAME = 'Trainer'
HOST_NAME = socket.gethostname()
SAVE_PATH = '{}/model'.format(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = '{}/config.ini'.format(os.path.dirname(os.path.abspath(__file__)))

config = ConfigParser()
config.read(CONFIG_FILE)
session = FuturesSession()
rd = redis.StrictRedis(host=config['REDIS']['HOST'],
                       port=config['REDIS']['PORT'],
                       password=config['REDIS']['PASS'], db=0)
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'],
                         'secret': config['BINANCE']['SECRET']})


def make_keras_picklable():
    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__
    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


class Symbol(object):
    def __init__(self, obj):
        self.__dict__ = json.loads(json.dumps(obj))


def log(text):
    msg = '{} {} {} {}'.format(time.strftime("%d/%m/%Y %H:%M"), HOST_NAME, BOT_NAME, text)
    url = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}&parse_mode=markdown' \
        .format(config['TELEGRAM']['BOT'], config['TELEGRAM']['CHAT'], msg)
    session.get(url)
    print(msg)
    return


def open_model(new=False):
    name = LSTM_MODEL
    model = None
    filename = '{}/{}.h5'.format(SAVE_PATH, name)
    log('load LSTM model from file: {}'.format(filename))
    try:
        model = load_model(filename)
        log('load LSTM model from file success: {}'.format(name))
    except Exception as e:
        log('load LSTM model from file error: {}'.format(str(e)))
        log('load LSTM model from redis: {}'.format(name))
        try:
            model = pickle.loads(zlib.decompress(rd.get(name)), encoding='latin1')
            log('load LSTM model from redis success: {}'.format(name))
        except Exception as e:
            log('load LSTM model from redis error: {}'.format(str(e)))
    new_model = Sequential()
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True,
                       batch_input_shape=(LSTM_SIZE * 100, STEP_SIZE, FEATURE_SIZE)))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(FEATURE_SIZE, stateful=True))
    new_model.add(Dense(FEATURE_SIZE, activation='linear'))
    new_model.add(Dense(1))
    new_model.compile(optimizer='adagrad', metrics=['accuracy'], loss='mse')
    if not new and model is not None:
        new_model.set_weights(model.get_weights())
    else:
        log('create LSTM model success: {}'.format(name))
    return new_model


def save_model(model):
    new_model = Sequential()
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True,
                       batch_input_shape=(1, STEP_SIZE, FEATURE_SIZE)))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(FEATURE_SIZE, stateful=True))
    new_model.add(Dense(FEATURE_SIZE, activation='linear'))
    new_model.add(Dense(1))
    new_model.compile(optimizer='adagrad', metrics=['accuracy'], loss='mse')  # adagrad
    new_model.set_weights(model.get_weights())
    name = LSTM_MODEL
    filename = '{}/{}.h5'.format(SAVE_PATH, name)
    try:
        log('save LSTM model to file: {}'.format(filename))
        new_model.save(filename)
        log('save LSTM model to file success')
    except Exception as e:
        log('save LSTM model to file error: {}'.format(str(e)))
    try:
        log('save LSTM model to redis: {}'.format(name))
        rd.set(name, zlib.compress(pickle.dumps(new_model)))
        log('save LSTM model to redis success')
    except Exception as e:
        log('save LSTM model to redis error: {}'.format(str(e)))


def symbol_data(symbol, time_frame):
    data = exchange.fetch_ohlcv(symbol, time_frame)
    df = DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('time')
    df.replace({0: np.nan}, inplace=True)
    df['price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df = df.assign(**{'volatility': lambda x: (x['high'] - x['low']) / x['open']})
    df = df.assign(**{'convergence': lambda x: (x['open'] - x['close']) / (x['high'] - x['low'])})
    df = df.assign(**{'predisposition': lambda x: 1 - 2 * (x['high'] - x['close']) / (x['high'] - x['low'])})
    df.dropna(axis=0, how='any', inplace=True)
    sc = MinMaxScaler(feature_range=(-1, 1))
    na = sc.fit_transform(df[['price_change', 'volume_change', 'volatility', 'convergence', 'predisposition']])
    return na, na[:, 0]


def save_data():
    exchange.load_markets(reload=True)
    train_x = []
    train_y = []
    for time_frame in TIME_FRAMES:
        log('start collect data with time frame: {}'.format(time_frame))
        symbols = exchange.symbols
        shuffle(symbols)
        for symbol in symbols:
            log('start collect data from symbol: {}'.format(symbol))
            input_data, output_data = symbol_data(symbol, time_frame)
            if len(input_data) == 0:
                continue
            for i in range(len(input_data) - STEP_SIZE - 1):
                train_x.append(input_data[i:i + STEP_SIZE])
                train_y.append(output_data[i + STEP_SIZE])
    length = int(len(train_x) // (LSTM_SIZE * 100) * (LSTM_SIZE * 100))
    train_x = train_x[-length:]
    train_y = train_y[-length:]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    np.save('{}/train_x.npy'.format(SAVE_PATH), train_x)
    np.save('{}/train_y.npy'.format(SAVE_PATH), train_y)
    log('save train data success')


def train():
    model = open_model()
    train_x = np.load('{}/train_x.npy'.format(SAVE_PATH))
    train_y = np.load('{}/train_y.npy'.format(SAVE_PATH))
    log('train LSTM model with data length: %g' % len(train_x))
    model.fit(train_x, train_y, epochs=LSTM_SIZE * 10, batch_size=LSTM_SIZE * 100, verbose=1)
    log('train LSTM model success')
    save_model(model)


def main():
    log('*{} started*'.format(BOT_NAME))
    make_keras_picklable()
    save_data()
    train()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import ccxt
from configparser import ConfigParser
import json
import os
import pickle
import redis
import socket
import tempfile
import time
import threading
import zlib
import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series
from requests_futures.sessions import FuturesSession
from sklearn.preprocessing import MinMaxScaler
import keras.models


BLACK_LIST = ['BNB']
CRON_TIME = 15
TA_TIME_FRAME = '15m'
ML_TIME_FRAME = '1h'  # '1h', '4h', '1d'
STEP_SIZE = 5
BOT_NAME = 'Detector'
HOST_NAME = socket.gethostname()
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


def crossed(series1, series2, direction=None):
    if isinstance(series1, np.ndarray):
        series1 = Series(series1)
    if isinstance(series2, int) or isinstance(series2, float) or isinstance(series2, np.ndarray):
        series2 = Series(index=series1.index, data=series2)
    if direction is None or direction == "above":
        above = Series((series1 > series2) & (
                series1.shift(1) <= series2.shift(1)))
    if direction is None or direction == "below":
        below = Series((series1 < series2) & (
                series1.shift(1) >= series2.shift(1)))
    if direction is None:
        return above or below
    return above if direction is "above" else below


def crossed_above(series1, series2):
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    return crossed(series1, series2, "below")


def get_buy_price(symbol):
    buy_price = rd.get('{}_buy'.format(symbol.symbol))
    if buy_price is None:
        return 0
    else:
        return float(buy_price)


def stop_detect(symbol, quote_change):
    time.sleep(1)
    try:
        if quote_change > 10:
            log('{} STOP SELL quote up'.format(symbol.symbol))
            rd.publish('stop_sell', symbol.symbol)
            return True
        if quote_change < -5:
            log('{} STOP SELL quote down'.format(symbol.symbol))
            rd.publish('stop_sell', symbol.symbol)
            return True
        if exchange.fetch_ticker(symbol.symbol)['change'] < 0:
            log('{} STOP SELL symbol down'.format(symbol.symbol))
            rd.publish('stop_sell', symbol.symbol)
            return True
        if exchange.fetch_balance()['free'][symbol.base] > symbol.limits['amount']['min']:
            buy_price = get_buy_price(symbol)
            sell_price = exchange.fetch_order_book(symbol.symbol)['asks'][0][0]
            if buy_price > 0 and (sell_price / buy_price - 1) * 100 > 3:
                log('{} STOP SELL take profit'.format(symbol.symbol))
                rd.publish('stop_sell', symbol.symbol)
                return True
            if buy_price > 0 and (sell_price / buy_price - 1) * 100 < -3:
                log('{} STOP SELL stop loss'.format(symbol.symbol))
                rd.publish('stop_sell', symbol.symbol)
                return True
    except Exception as e:
        log('{} error: {}'.format(symbol.symbol, str(e)))
        time.sleep(10)
    return False


def hf_detect(symbol):
    try:
        order_book = exchange.fetch_order_book(symbol.symbol)
        buy_price = round(order_book['bids'][0][0] + symbol.limits['price']['min'], symbol.precision['price'])
        sell_price = round(order_book['asks'][0][0] - symbol.limits['price']['min'], symbol.precision['price'])
        if (sell_price / buy_price - 1) * 100 > 1.7:
            log('{} HF BUY'.format(symbol.symbol))
            rd.publish('hf_buy', symbol.symbol)
            return True
        return False
    except Exception as e:
        log('{} error: {}'.format(symbol.symbol, str(e)))
        time.sleep(30)
    return False


def ta_detect(symbol):
    try:
        data = exchange.fetch_ohlcv(symbol.symbol, TA_TIME_FRAME)
        df = DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index('time', inplace=True, drop=True)
        df['rsi'] = ta.RSI(df)
        df['adx'] = ta.ADX(df)
        df['plus_di'] = ta.PLUS_DI(df)
        df['minus_di'] = ta.MINUS_DI(df)
        df['fastd'] = ta.STOCHF(df)['fastd']
        df.loc[
            (
                    (df['rsi'] < 35) &
                    (df['fastd'] < 35) &
                    (df['adx'] > 30) &
                    (df['plus_di'] > 0.5)
            ) |
            (
                    (df['adx'] > 65) &
                    (df['plus_di'] > 0.5)
            ),
            'buy'] = 1
        df.loc[
            (
                    (
                            (crossed_above(df['rsi'], 70)) |
                            (crossed_above(df['fastd'], 70))
                    ) &
                    (df['adx'] > 10) &
                    (df['minus_di'] > 0)
            ) |
            (
                    (df['adx'] > 70) &
                    (df['minus_di'] > 0.5)
            ),
            'sell'] = 1
        buy_signal, sell_signal = df.iloc[-1]['buy'], df.iloc[-1]['sell']
        if buy_signal == 1:
            log('{} TA BUY'.format(symbol.symbol))
            rd.publish('ta_buy', symbol.symbol)
            return True
        elif sell_signal == 1:
            log('{} TA SELL'.format(symbol.symbol))
            rd.publish('ta_sell', symbol.symbol)
            return True
        return False
    except Exception as e:
        log('{} error: {}'.format(symbol.symbol, str(e)))
        time.sleep(30)
    return False


def ml_detect(symbol, model):
    try:
        data_base = exchange.fetch_ohlcv(symbol.symbol, ML_TIME_FRAME)
        df = DataFrame(data_base, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
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
        input_data = sc.fit_transform(df[['price_change', 'volume_change', 'volatility', 'convergence', 'predisposition']])
        output_data = input_data[:, 0]
        mean = np.mean(output_data, axis=0)
        last_change = output_data[-1] - mean
        predict_change = model.predict(np.array([input_data[-STEP_SIZE:]]), batch_size=1)[0][0] - mean
        if last_change < 0 < .2 < predict_change:
            log('{} ML BUY'.format(symbol.symbol))
            rd.publish('ml_buy', symbol.symbol)
            return True
        elif last_change > 0 > -.1 > predict_change:
            log('{} ML SELL'.format(symbol.symbol))
            rd.publish('ml_sell', symbol.symbol)
            return True
        return False
    except Exception as e:
        log('{} error: {}'.format(symbol.symbol, str(e)))
        time.sleep(30)
    return False


def hf_run(symbols, start_time):
    time.sleep(CRON_TIME)
    run_time = time.time()
    while run_time - start_time < 60 * CRON_TIME:
        for symbol in symbols:
            hf_detect(symbol)
            time.sleep(1)
            run_time = time.time()
            if run_time - start_time > 60 * CRON_TIME:
                break


def ta_run(symbols, start_time):
    time.sleep(CRON_TIME * 2)
    run_time = time.time()
    while run_time - start_time < 60 * CRON_TIME:
        for symbol in symbols:
            ta_detect(symbol)
            time.sleep(5)
            run_time = time.time()
            if run_time - start_time > 60 * CRON_TIME:
                break


def ml_run(symbols, start_time):
    time.sleep(CRON_TIME * 3)
    try:
        model = pickle.loads(zlib.decompress(rd.get('lstm')), encoding='latin1')
        run_time = time.time()
        while run_time - start_time < 60 * CRON_TIME:
            for symbol in symbols:
                ml_detect(symbol, model)
                time.sleep(10)
                run_time = time.time()
                if run_time - start_time > 60 * CRON_TIME:
                    break
    except Exception as e:
        log('Error: {}'.format(str(e)))


def main():
    log('*{} started*'.format(BOT_NAME))
    make_keras_picklable()
    start_time = time.time()
    p = rd.pubsub(ignore_subscribe_messages=True)
    p.subscribe('stop_sell', 'hf_buy', 'ta_buy', 'ta_sell', 'ml_buy', 'ml_sell')
    exchange.load_markets(reload=True)
    quote_change = exchange.fetch_ticker('{}/USDT'.format(config['CONFIG']['QUOTE']))['change']
    symbols = []
    for key in exchange.symbols:
        symbol = Symbol(exchange.market(key))
        if symbol.quote == config['CONFIG']['QUOTE'] and symbol.base not in BLACK_LIST:
            if not stop_detect(symbol, quote_change):
                symbols.append(symbol)
    if len(symbols) > 0:
        if config['CONFIG']['HF'] == 'yes':
            thread1 = threading.Thread(target=hf_run, args=(symbols, start_time))
            thread1.start()
        if config['CONFIG']['TA'] == 'yes':
            thread2 = threading.Thread(target=ta_run, args=(symbols, start_time))
            thread2.start()
        if config['CONFIG']['ML'] == 'yes':
            thread3 = threading.Thread(target=ml_run, args=(symbols, start_time))
            thread3.start()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import bz2
import ccxt
from configparser import ConfigParser
import os
import plotly
import redis
import socket
import time
import plotly.graph_objs as go
from requests_futures.sessions import FuturesSession

QUOTE_ASSET = 'ETH'
BLACK_LIST = ['BNB']
TIME_FRAME = '5m'
LOOK_BACK = 5
BOT_NAME = 'Reporter'
HOST_NAME = socket.gethostname()
CONFIG_FILE = '{}/config.ini'.format(os.path.dirname(os.path.abspath(__file__)))
HTML_FILE = '{}/balance.html'.format(os.path.dirname(os.path.abspath(__file__)))

config = ConfigParser()
config.read(CONFIG_FILE)
session = FuturesSession()
rd = redis.StrictRedis(host=config['REDIS']['HOST'],
                       port=config['REDIS']['PORT'],
                       password=config['REDIS']['PASS'], db=0)
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'],
                         'secret': config['BINANCE']['SECRET']})


def log(text):
    msg = '{} {} {} {}'.format(time.strftime("%d/%m/%Y %H:%M"), HOST_NAME, BOT_NAME, text)
    url = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}&parse_mode=markdown' \
        .format(config['TELEGRAM']['BOT'], config['TELEGRAM']['CHAT'], msg)
    session.get(url)
    print(msg)
    return


def to_quote(base, amount):
    if base == QUOTE_ASSET:
        return 1, amount, 0
    symbol = base + '/' + QUOTE_ASSET
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last'], ticker['last'] * amount, ticker['change']
    except Exception as e:
        return 0, 0, 0


def to_usdt(total):
    btc = 'BTC/USDT'
    quote = QUOTE_ASSET + '/USDT'
    btc_ticker = exchange.fetch_ticker(btc)
    quote_ticker = exchange.fetch_ticker(quote)
    return btc_ticker['last'], quote_ticker['last'], quote_ticker['last'] * total


def set_history():
    balance = exchange.fetch_balance()['total']
    date = time.strftime("%d/%m/%Y %H:%M")
    quote_total = 0
    text = 'account balance:  \n'
    text += 'Total balance in %s:  \n' % QUOTE_ASSET
    text += '%s amount: %g' % (QUOTE_ASSET, balance[QUOTE_ASSET]) + '  \n'
    b = '%s:%g ' % (QUOTE_ASSET, balance[QUOTE_ASSET])
    for key in sorted(balance.keys()):
        if key == QUOTE_ASSET:
            continue
        amount = balance[key]
        if amount > 0:
            price, value, change = to_quote(key, balance[key])
            text += '%s amount: %g price: %g value: %g change: %.2f%%' % (key, amount, price, value, change) + '  \n'
            b += '%s:%g ' % (key, value)
            quote_total += value
            time.sleep(0.1)
    rd.set('altcoin_value', quote_total)
    quote_total += balance[QUOTE_ASSET]
    text += 'Total in %s: `%g`' % (QUOTE_ASSET, quote_total)
    log(text)
    btc_price, quote_price, quote_value = to_usdt(quote_total)
    s = '%s,%g,%g,%g,%g,%s;' % (date, btc_price, quote_price, quote_total, quote_value, b[:-1])
    history = rd.get('history')
    if history is None:
        history = ''
    else:
        history = bz2.decompress(history).decode('latin1')
    history += s
    rd.set('history', bz2.compress(history.encode()))


def get_history():
    arr0 = []
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    history = rd.get('history')
    if history is not None:
        history = bz2.decompress(history).decode('latin1')
        for s in history.split(';'):
            a = s.split(',')
            if len(a) == 6:
                arr0.append(a[0])
                arr1.append(a[1])
                arr2.append(a[2])
                arr3.append(a[3])
                arr4.append(a[4])
                arr5.append(a[5])
    return arr0, arr1, arr2, arr3, arr4, arr5


def rd_get(key):
    value = rd.get(key)
    if value is None:
        value = 0
    else:
        value = float(value)
    return value


def set_total():
    date = time.strftime("%d/%m/%Y %H:%M")
    hf_buy = rd_get('HF_buy_total')
    hf_sell = rd_get('HF_sell_total')
    ta_buy = rd_get('TA_buy_total')
    ta_sell = rd_get('TA_sell_total')
    ml_buy = rd_get('ML_buy_total')
    ml_sell = rd_get('ML_sell_total')
    stop = rd_get('SS_sell_total')
    fee = rd_get('fee_total')
    altcoin_value = rd_get('altcoin_value')
    profit = -hf_buy + hf_sell - ta_buy + ta_sell - ml_buy + ml_sell + stop - fee
    text = 'trade result:  \n'
    text += 'Total HF: buy: %g sell: %g profit: %g  \n' % (hf_buy, hf_sell, hf_sell - hf_buy)
    text += 'Total TA: buy: %g sell: %g profit: %g  \n' % (ta_buy, ta_sell, ta_sell - ta_buy)
    text += 'Total ML: buy: %g sell: %g profit: %g  \n' % (ml_buy, ml_sell, ml_sell - ml_buy)
    text += 'Total SS: sell: %g  \n' % stop
    text += 'Total fee: %g  \n' % fee
    text += 'Total altcoin value: %g  \n' % altcoin_value
    text += 'Total profit: `%g`' % profit
    log(text)
    s = '%s,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g;' % \
        (date, -hf_buy, hf_sell, -ta_buy, ta_sell, -ml_buy, ml_sell, stop, -fee, profit, altcoin_value)
    total = rd.get('total')
    if total is None:
        total = ''
    else:
        total = bz2.decompress(total).decode('latin1')
    total += s
    rd.set('total', bz2.compress(total.encode()))


def get_total():
    arr0 = []
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    arr10 = []
    total = rd.get('total')
    if total is not None:
        total = bz2.decompress(total).decode('latin1')
        for s in total.split(';'):
            a = s.split(',')
            if len(a) >= 10:
                arr0.append(a[0])
                arr1.append(a[1])
                arr2.append(a[2])
                arr3.append(a[3])
                arr4.append(a[4])
                arr5.append(a[5])
                arr6.append(a[6])
                arr7.append(a[7])
                arr8.append(a[8])
                arr9.append(a[9])
            if len(a) == 11:
                arr10.append(a[10])
    return arr0, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10


def gen_chart():
    date, btc_price, eth_price, eth_total, usdt_total, balance = get_history()
    btc_price_trace = go.Scatter(x=date, y=btc_price, mode='lines', name='BTC Price',
                                 line=dict(width=1.5, color='blue', shape='spline'))
    eth_price_trace = go.Scatter(x=date, y=eth_price, mode='lines', name='ETH Price',
                                 line=dict(width=1.5, color='green', shape='spline'))
    eth_total_trace = go.Scatter(x=date, y=eth_total, mode='lines', name='ETH Total',
                                 line=dict(width=3, color='red', shape='spline'), yaxis='y2')
    usdt_total_trace = go.Scatter(x=date, y=usdt_total, mode='lines', name='USDT Total',
                                  line=dict(width=3, color='orange', shape='spline'))
    date, hf_buy, hf_sell, ta_buy, ta_sell, ml_buy, ml_sell, stop, fee, profit, altcoin_value = get_total()
    hf_buy_trace = go.Scatter(x=date, y=hf_buy, mode='none', fill='tonexty', name='HF Buy', xaxis='x3', yaxis='y3')
    hf_sell_trace = go.Scatter(x=date, y=hf_sell, mode='none', fill='tonexty', name='HF Sell', xaxis='x3', yaxis='y3')
    ta_buy_trace = go.Scatter(x=date, y=ta_buy, mode='none', fill='tonexty', name='TA Buy', xaxis='x3', yaxis='y3')
    ta_sell_trace = go.Scatter(x=date, y=ta_sell, mode='none', fill='tonexty', name='TA Sell', xaxis='x3', yaxis='y3')
    ml_buy_trace = go.Scatter(x=date, y=ml_buy, mode='none', fill='tonexty', name='ML Buy', xaxis='x3', yaxis='y3')
    ml_sell_trace = go.Scatter(x=date, y=ml_sell, mode='none', fill='tonexty', name='ML Sell', xaxis='x3', yaxis='y3')
    stop_trace = go.Scatter(x=date, y=stop, mode='none', fill='tonexty', name='Stop Sell', xaxis='x3', yaxis='y3')
    fee_trace = go.Scatter(x=date, y=fee, mode='none', fill='tonexty', name='Fee', xaxis='x3', yaxis='y3')
    altcoin_value_trace = go.Scatter(x=date, y=altcoin_value, mode='none', fill='tonexty', name='Altcoin Value', xaxis='x3', yaxis='y3')
    profit_trace = go.Scatter(x=date, y=profit, mode='lines', name='Profit',
                              line=dict(width=3, color='red', shape='spline'), xaxis='x3', yaxis='y3')
    labels = []
    values = []
    pairs = balance[-1].split(' ')
    for pair in pairs:
        labels.append(pair.split(':')[0])
        values.append(pair.split(':')[1])
    pie = go.Pie(labels=labels, values=values, domain={'x': [0.7, 1], 'y': [0, 0.45]})
    data = [hf_sell_trace, stop_trace, ta_sell_trace, ml_sell_trace, hf_buy_trace, ta_buy_trace, ml_buy_trace,
           altcoin_value_trace, fee_trace, profit_trace, btc_price_trace, eth_price_trace, eth_total_trace, usdt_total_trace, pie]
    layout = go.Layout(
        title='Auto Trade Account Balance',
        xaxis=dict(tickangle=45, domain=[0, 1], anchor='y'),
        yaxis=dict(title='USDT', domain=[0.55, 1]),
        yaxis2=dict(title='ETH', overlaying='y', side='right', domain=[0.55, 1], anchor='x'),
        xaxis3=dict(tickangle=45, domain=[0, 0.6], anchor='y3'),
        yaxis3=dict(title='ETH', domain=[0, 0.45]),

    )
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=HTML_FILE)


def main():
    log('*{} started*'.format(BOT_NAME))
    exchange.load_markets()
    set_history()
    set_total()
    gen_chart()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import ccxt
from configparser import ConfigParser
import json
import os
import redis
import socket
import threading
import time
from random import randint
from requests_futures.sessions import FuturesSession

CRON_TIME = 15
PANIC_COUNT = 5
BOT_NAME = 'Trader'
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


def add(key, value):
    total = rd.get(key)
    if total is None:
        total = 0
    else:
        total = float(total)
    total += value
    rd.set(key, total)


def save_buy_total(algo, symbol, amount, buy_price):
    rd.set('{}_buy'.format(symbol.symbol), buy_price)
    add('buy_total', amount * buy_price)
    add('{}_buy_total'.format(algo), amount * buy_price)
    add('fee_total', amount * buy_price * 0.005)


def save_sell_total(algo, symbol, amount, sell_price):
    rd.set('{}_sell'.format(symbol.symbol), sell_price)
    add('sell_total', amount * sell_price)
    add('{}_sell_total'.format(algo), amount * sell_price)
    add('fee_total', amount * sell_price * 0.005)


def get_buy_price(symbol):
    buy_price = rd.get('{}_buy'.format(symbol.symbol))
    if buy_price is None:
        return 0
    else:
        return float(buy_price)


def buy_bnb():
    symbol = 'BNB/{}'.format(config['CONFIG']['QUOTE'])
    last_price = exchange.fetch_ticker(symbol)['last']
    min_cost = exchange.market(symbol)['limits']['cost']['min']
    amount = int(min_cost / last_price)
    if exchange.fetch_balance()['free']['BNB'] < amount:
        exchange.create_market_buy_order(symbol, amount)
        log('buy BNB for fee, amount: %g' % amount)


def price_calculate(symbol):
    order_book = exchange.fetch_order_book(symbol.symbol)
    buy_price = round(order_book['bids'][0][0] + symbol.limits['price']['min'] * randint(2, 5), symbol.precision['price'])
    sell_price = round(order_book['asks'][0][0] - symbol.limits['price']['min'] * randint(2, 5), symbol.precision['price'])
    return buy_price, sell_price


def order_status(order_id, symbol):
    order = exchange.fetch_order(order_id, symbol.symbol)
    status = order['status']
    filled = order['filled']
    remaining = order['remaining']
    if status == 'open' and filled > 0:
        status = 'parted'
    return status, filled, remaining


def buy(algo, symbol, panic=0):
    if symbol is None:
        return
    panic += 1
    if panic > PANIC_COUNT:
        return
    try:
        buy_price, sell_price = price_calculate(symbol)
        amount = round(int(float(config['CONFIG']['BUDGET']) / buy_price / symbol.limits['amount']['min'])
                       * symbol.limits['amount']['min'], symbol.precision['amount'])
        if amount < symbol.limits['amount']['min']:
            return
        if amount * buy_price < symbol.limits['cost']['min']:
            return
        balance = exchange.fetch_balance()
        if balance['total'][symbol.base] >= amount:
            return
        log('%s %s buy amount:%.8f price:%.8f total:%.8f'
            % (symbol.symbol, algo, amount, buy_price, amount * buy_price))
        order = exchange.create_limit_buy_order(symbol.symbol, amount, buy_price)
        time.sleep(1)
        order_id = order['id']
        panic_buy = 0
        while True:
            status, filled, remaining = order_status(order_id, symbol)
            if status == 'open':
                panic_buy += 1
                if panic_buy > PANIC_COUNT:
                    exchange.cancel_order(order_id, symbol.symbol)
                    time.sleep(1)
                    buy(algo, symbol, panic)
                    break
                else:
                    time.sleep(1)
                    continue
            elif status == 'parted':
                log('%s %s buy partially filled, amount:%.8f' % (symbol.symbol, algo, filled))
                panic_buy += 1
                if panic_buy > PANIC_COUNT:
                    exchange.cancel_order(order_id, symbol.symbol)
                    save_buy_total(algo, symbol, filled, buy_price)
                else:
                    time.sleep(1)
                    continue
            elif status == 'closed':
                log('%s %s buy filled, amount:%.8f' % (symbol.symbol, algo, amount))
                save_buy_total(algo, symbol, amount, buy_price)
            else:
                log('%s %s buy failed, status:%s' % (symbol.symbol, algo, status))
                exchange.cancel_order(order_id, symbol.symbol)
            if algo == 'HF':
                sell(algo, symbol)
            break
    except Exception as e:
        log('{} error: {}'.format(symbol.symbol, str(e)))
    return


def sell(algo, symbol):
    if symbol is None:
        return
    try:
        buy_price, sell_price = price_calculate(symbol)
        balance = exchange.fetch_balance()
        amount = round(int(balance['free'][symbol.base] / symbol.limits['amount']['min'])
                       * symbol.limits['amount']['min'], symbol.precision['amount'])
        if amount < symbol.limits['amount']['min']:
            return
        if amount * sell_price < symbol.limits['cost']['min']:
            return
        log('%s %s sell amount:%.8f price:%.8f total:%.8f'
            % (symbol.symbol, algo, amount, sell_price, amount * sell_price))
        order = exchange.create_limit_sell_order(symbol.symbol, amount, sell_price)
        time.sleep(1)
        order_id = order['id']
        panic_sell = 0
        while True:
            status, filled, remaining = order_status(order_id, symbol)
            if status == 'open':
                panic_sell += 1
                if panic_sell > PANIC_COUNT:
                    exchange.cancel_order(order_id, symbol.symbol)
                    time.sleep(1)
                    sell(algo, symbol)
                else:
                    time.sleep(1)
                    continue
            elif status == 'parted':
                log('`%s %s sell partially filled, amount:%.8f`' % (symbol.symbol, algo, filled))
                panic_sell += 1
                if panic_sell > PANIC_COUNT:
                    exchange.cancel_order(order_id, symbol.symbol)
                    save_sell_total(algo, symbol, filled, sell_price)
                    buy_price = get_buy_price(symbol)
                    if buy_price > 0:
                        log('`%s %s possible profit: %.8f %.2f%%`' %
                            (symbol.symbol, algo, (sell_price - buy_price) * filled, (sell_price / buy_price - 1) * 100))
                    time.sleep(1)
                    sell(algo, symbol)
                else:
                    time.sleep(1)
                    continue
            elif status == 'closed':
                log('`%s %s sell filled, amount:%.8f`' % (symbol.symbol, algo, amount))
                save_sell_total(algo, symbol, amount, sell_price)
                buy_price = get_buy_price(symbol)
                if buy_price > 0:
                    log('`%s %s possible profit: %.8f %.2f%%`' %
                        (symbol.symbol, algo, (sell_price - buy_price) * amount, (sell_price / buy_price - 1) * 100))
                    if algo == 'HF' and (sell_price / buy_price - 1) * 100 > 1:
                        buy(algo, symbol)
            else:
                log('%s sell failed, status:%s' % (symbol.symbol, status))
                exchange.cancel_order(order_id, symbol.symbol)
            break
    except Exception as e:
        log('{} error: {}'.format(symbol.symbol, str(e)))
    return


def handler(message):
    symbol = Symbol(exchange.market(message['data'].decode('latin1')))
    algo = message['channel'].decode('latin1').split('_')[0].upper()
    target = message['channel'].decode('latin1').split('_')[1]
    if target == 'buy':
        thread = threading.Thread(target=buy, args=(algo, symbol))
        thread.start()
    elif target == 'sell':
        thread = threading.Thread(target=sell, args=(algo, symbol))
        thread.start()


def main():
    log('*{} started*'.format(BOT_NAME))
    start_time = time.time()
    exchange.load_markets(reload=True)
    buy_bnb()
    p = rd.pubsub(ignore_subscribe_messages=True)
    p.subscribe(**{'stop_sell': handler})
    p.subscribe(**{'hf_buy': handler})
    p.subscribe(**{'ta_buy': handler})
    p.subscribe(**{'ta_sell': handler})
    p.subscribe(**{'ml_buy': handler})
    p.subscribe(**{'ml_sell': handler})
    current_time = time.time()
    while current_time - start_time < 60 * CRON_TIME:
        p.get_message()
        time.sleep(1)
        current_time = time.time()
    for order in exchange.fetch_open_orders():
        exchange.cancel_order(order['id'], order['symbol'])


if __name__ == "__main__":
    main()

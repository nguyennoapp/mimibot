# mimibot
 Cryptocurrencies trading bot

# Features:
- Support multi exchanges by using ccxt library (tested on Binance only)
- Auto scan and select symbol to trade
- Three type of detecting signal:
    + HF: high frequency
    + TA: technical analysis
    + ML: machine learning    
    (see more on code, there's no comments on code but it's readable though)
- Real-time tracking Telegram bot
- Statistic report web page

# Installation:
- Create Binance account
- Create Binance API with trading permission
- Deposit some
- Install all needed python packages by using pip
- Build ta-lib and tensor-flow from sources (talib python and keras are just api wrapper built on top of them)
- Install redis
- Edit config.ini
- Add to crontab:
    + trainer.py: once a day
    + reporter.py: once an hour
    + detector.py and trader.py: each 15 minutes

# Is it profitable?
- Sometimes it is, sometimes it's not, I don't know why!? Let's improve the code.

# Reference:
- https://github.com/yasinkuyu/binance-trader
- https://github.com/gcarq/freqtrade
- and others..

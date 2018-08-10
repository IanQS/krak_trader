# Setup

1) [Slack Join Link](https://join.slack.com/t/krakentrader/shared_invite/enQtNDEwODUwOTI5MjIwLWNhZDk3NmIzM2Y4YjAwYTcxMDI5ZWIxNmMzNGIyYzE4ZjY0MzA1OGFmOTExYmY0Yjk3ZjI2ZDAxMmY3NWQwN2U)

2) Before starting, create a `src/constants.py` file that contains:


```asciidoc
__STORAGE_PATH = 'BASE_PATH/news_data/{}'

# where base_path is where you want to store the scraped data to

KRAKEN_PATH = __STORAGE_PATH.format("kraken") + "/{}"
NEWS_PATH = __STORAGE_PATH.format("news") + "/{}"

```

# Contributing

1) Create a new branch to work on, create a PR and we'll merge it in AFTER its been reviewed

2) Don't EVER commit your API key to this as we'll try and make it public so your contributions will be listed on your profile


# Data

- will contain kraken price data and news data

[Dropbox Link](https://www.dropbox.com/sh/11ln04vn0n6ojuv/AABYIzzrp5UEvLKGhYLxnMISa?dl=0)

# Guidelines

1) Inherit from `base_scraper` as it handles all the paths, and settings

2) Register your function to `registered.py`, so that `launch_scrapers` can kick off all the scrapers easily

3) All the news scrapers will be threaded. Prepare accordingly by:
    - starting the connection within `init` AFTER all initialization has occurred

    - handling errors gracefully. Errors should be logged to a file (append mode)
    
    - closing the connection if using an api
    
# Notes

1. [NewsApiClient - API Wrapper](https://newsapi.org/docs/client-libraries/python)
    - query takes multiple arguments, but takes the intersection of keywords, not union
    
2. [Crypto Coins News](crypto_coins_news.py) should serve as a reference for:
    - features to serve
    
    - how to interact with the [base class](base_scraper.py)
    
    - how to interact with the NewsApiClient  

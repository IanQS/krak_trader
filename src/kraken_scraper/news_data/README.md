# Guidelines

1) Inherit from `base_scraper` as it handles all the paths, and settings

2) Register your function to `registered.py`, so that `launch_scrapers` can kick off all the scrapers easily

3) All the news scrapers will be multiprocessed. Prepare accordingly by:
    - starting the connection within `init` AFTER all initialization has occurred

    - handling errors gracefully. Errors should be logged to a file (append mode)
    
    - closing the connection if using an api. 
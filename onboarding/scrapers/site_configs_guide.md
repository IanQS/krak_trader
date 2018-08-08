# How to determine if a site should use Selenium to be scraped.
### Introduction
Selenium has a lot of overhead during scraping as its main purpose is to automate web testing.
So it would be best to avoid using Selenium as much as possible and use Python's requests library instead.
However, since many websites use JavaScript to push their content, we still require Selenium to render content from them.

### How to determine if a site requires Selenium
1. Open your web browser and navigate to an article from the site.
2. Inspect the source HTML.
3. Fire up your Python interpreter and make a query to the same URL using the `requests` lib.
    ```
    >>> import requests
    >>> site = requests.get(url)
    >>> site.text
    ```
4. Compare the HTML returned from your browser to the HTML returned from `requests`.
5. * If you are **ABLE** to locate the content of the article returned by `requests`, then all is good and you can bask in the speed up you have gained.
   * If you are **NOT ABLE** to locate the content of the article returned by `requests` *or* some other error occurs, then you're doomed to using Selenium to painfully scrape content from that site for all eternity.
6. Proceed to configure the settings for that site in `site_configs.py`.

# Data format in site\_configs.py
The file `site_configs.py` is used to inject site specific settings into the scrapers.
We will use the settings for crypto-coins-news as an example for how to set up a site:
```
"crypto-coins-news": {
    "selenium": True,
    "api": True,
    "html_tag": "div",
    "tag_attributes": {
        "class": "entry-content"
    }
}
```
| Field | Desription 
|---| ---
|*crypto-coins-news* |Site id as stored in the NewsApi, check [here](https://newsapi.org/docs/endpoints/sources) for the list of ids. If the site is not part of NewsApi, then name it using all lowercase letters, with hyphens in between words.
|*selenium*          | Set to True if the website requires selenium to be scraped, usually websites whose content are JavaScript driven.
|*api*               | Set to True if the website is covered under the NewsApi.
|*html\_tag*         | The html tag to be scraped for content. Most commonly used is `div`. Try to localize the content as much as possible, but don't worry about capturing garbage along with actual content, it can be filtered out later. 
|*tag\_attributes*  | The attributes for *html\_tag*. HTML tags have attributes along with them to help with their identification, e.g. `<div class="content-here" id="most-important-content" > ... </div>`. If this is the tag you want, put its attributes as follows: `"tag_attributes" : { "class": "content-here", "id"   : "most-important-content"}`

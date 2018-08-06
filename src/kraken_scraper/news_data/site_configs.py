"""
site_config
    - settings for newsapi based websites to be injected into news_api_scraper.

Data format explanation using "crypto-coins-news" as an example:
    "crypto-coins-news" : Site id as stored in the newsapi
    "selenium"          : True if the website requires selenium to be scraped, usually websites whose content are javascript driven
    "api"               : True if the website is covered under newsapi
    "html_tag"          : The html tag to be scraped for content
    "tag_attributes"    : The attributes for "html_tag"

Author: Bryan Q.
"""

SITE_CONF = {
    "crypto-coins-news": {
        "selenium": True,
        "api": True,
        "html_tag": "div",
        "tag_attributes": {
            "class": "entry-content"
        }
    },

    "coin-telegraph": {
        "selenium": True,
        "api": False,
        "html_tag": "div",
        "tag_attributes": {
            "class" :"post-content"
        }
    },

    "techcrunch": {
        "selenium": True,
        "api": True,
        "html_tag": "div",
        "tag_attributes": {
            "class": "article-content"
        }
    }
}

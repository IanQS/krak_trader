"""
site_config
    - settings for newsapi based websites to be injected into news_api_scraper.

Author: Bryan Q.

Notes:
    Check "onboarding/scrapers/site_configs_guide.md" for more details.
"""

SITE_CONF = {
    "crypto-coins-news": {
        "api": True,
        "content-xpath": "//div[@class='entry-content']"
    },

    "coin-telegraph": {
        "api": False,
        "content-xpath": "//div[@class='post-content']"
    },

    "techcrunch": {
        "api": True,
        "content-xpath": "//div[@class='article-content']"
    },

# bloomberg blocks bots lol
#    "bloomberg": {
#        "api": True,
#        "content-xpath": "//div[@class'body-copy-v2 fence-body']"
#    },

# business insider changes their HTML based on location, below is US
    "business-insider": {
        "api": True,
        "content-xpath": "//section[@data-post-type='post']/div/div[2]"
    },

# NewsApi returns some dead articles
    "the-new-york-times": {
        "api": True,
        "content-xpath": "//article[@id='story']"
    }
}

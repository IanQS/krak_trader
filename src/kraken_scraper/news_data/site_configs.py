"""
site_config
    - settings for newsapi based websites to be injected into news_api_scraper.

Author: Bryan Q.

Notes:
    Check "onboarding/scrapers/site_configs_guide.md" for more details.
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

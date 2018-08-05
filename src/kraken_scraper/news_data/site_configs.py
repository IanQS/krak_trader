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
        "api": False,
        "html_tag": "div",
        "tag_attributes": {
            "class": "article-content"
        }
    }
}

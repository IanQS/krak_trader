# Data format in site\_configs.py
The file `site_configs.py` is used to inject site specific settings into the scrapers.
We will use the settings for crypto-coins-news as an example for how to set up a site:
```
"crypto-coins-news": {
    "api": True,
	"content-xpath": "//div[@class='entry-content']"
}
```
| Field | Desription 
|---| ---
|*crypto-coins-news* | Site id as stored in the NewsApi, check [here](https://newsapi.org/docs/endpoints/sources) for the list of ids. If the site is not part of NewsApi, then name it using all lowercase letters, with hyphens in between words.
|*api*               | Set to True if the website is covered under the NewsApi.
|*content-xpath*     | In this case, the content we want is stored in `<div class='entry-content'> ... </div>` and the xpath we used will match all `div`'s with the attribute `class='entry-content'`, so be sure your xpath only matches one item; add more parents up to the root if necessary. Xpath also supports conditionals so if the target content is stored in `<potato id='1234' class='potatoes-are-fruits marshmellow'> ... </potato>`, then your query would be `//potato[@id='1234' and class='potatoes-are-fruits marshmellow'`.

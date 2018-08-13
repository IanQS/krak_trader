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
|*crypto-coins-news* | Site id as stored in the NewsApi, check [here](https://newsapi.org/docs/endpoints/sources) for the list of ids. If the site is not part of NewsApi, i.e. the field *api* is set to *False*, then name it using all lowercase letters, with hyphens in between words.
|*api*               | Set True if the website is covered under the NewsApi.
|*content-xpath*     | The xpath here captures all `div` html tags with the attribute `class='entry-content'`. Add more parents to the path, use conditionals, or use indices if your query matches multiple tags. Example: `<vegetable><potato id='1234' class='potatoes-are-fruits marshmellow'> ... </potato><potato id='1234'> ... </potato></vegetable>`, then your query would be `//potato[@id='1234' and @class='potatoes-are-fruits marshmellow']` to capture the first `potato` and `//potato[@id='1234'][2]` to capture the second `potato`.

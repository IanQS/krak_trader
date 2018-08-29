#NLP Overview

## Goals/Tasks
#### (Collected from various files below, loosely ordered by priority/dependence/difficulty)
1. **SpaCy / NER**: Determine which currency/ies are mentioned in an article. Possible forms to look out for: full spelling, ticker symbols, slang, poorly-cased, mis-spelled...
    1. Incorporate these aliases (e.g. "Bitcoin"/"bitcoin"/"BTC"/"btc") into some persistent data structure (likely in its own file)
1. **SpaCy / NER**: Bootstrap crypto-currency NER (tagging all such terms with a "MONEY" annotation) via IOB/BILUO from culled gold-standard data
    1. Make sure MONEY annotations include currency values (verify that MONEY applies to "25 USD", and not just to "USD")
    1. Examine the entities auto-classified by spacy as MONEY, and possibly augment ticker_aliases
1. **SpaCy**: train targeted language models based on scraped article texts and annotated currency terms
    1. Augment a pre-trained English model
    1. Bootstrap from a completely blank language model
    1. Evaluate these models against each other
        1. Choose metric(s) for this.
1. Create a semantic regressor to rate an article on a [-1,1] (or [0,1]) continuum of bad/good news for a currency
    1. Seek out quantitative as well as qualitative statements ("BTC fell by 1.3%"), ("things are looking good for coinCoin this quarter")
1. Evaluate predictions made in previous articles (Did prices actually rise/fall?). Two subtypes:
    1. Retrospective: talking about a price change at a time _before_ article publication date ("Dogecoin tanked last week")
    1. Predictive: talking about a price change at a time _after_ article publication date ("I think BTC will fall next week")
1. Use article analyses (see retrospective/predictive subtypes above) to score article authors by dependability
    1. How truthfully do they report past price changes?
    1. How reliably do they predict future price changes?  

## File structure
- **README.md** - this file
- **module_comparison/** - explorations of spaCy/CoreNLP modules/packages
  - **spacy_demo.py** - test out some features of the spaCy module

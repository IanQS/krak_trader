# module_comparison
### Explore benefits/drawbacks of spaCy & CoreNLP for NER and related parsing tasks

Currently leaning toward spaCy for bootstrapping targeted language models, but only CoreNLP has built-in sentiment analysis.

### TODO:
- Research sites for crypto news articles/blurbs/predictions
    - Determine which of these have APIs, and which need scraping
- Create a semantic regressor to rate an article on a [-1,1] continuum of good/bad news for a currency
    - Quantitative vs qualitative statements ("BTC fell by 1.3%"), ("this are looking good for coinCoin this quarter")
- NER: which currency/ies are mentioned in the article
- Evaluate predictions made in previous articles (did prices actually rise/fall)
    - use these to rate article authors by dependability
- build corpora of article rawtexts, and author scores
- ONGOING: Learn [markdown](https://guides.github.com/features/mastering-markdown/) and properly format this file (lol)
- Look at both predictive ("I think BTC will fall next week") and retrospective analyses ("dogecoin tanked last week")

### Helpful/educational links:
- Finance knowledge/education
    - [Learn to day-trade in a week](https://www.youtube.com/watch?v=GTtKLeDTCHo)
- [Tensorflow tutorial](https://bytegain.com/blog/getting-started-with-tensorflow-guide)
- NER
    - [SpaCy](https://spacy.io/)
        - [Train from custom annotations](https://spacy.io/usage/training)
            - See esp. the Basics/NER/TextClassification sections
                - Looks like they did a lot of the legwork for training classification models
        - [NER Training example](https://github.com/explosion/spacy/blob/master/examples/training/train_ner.py)
    - [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
        - The main advantage with this is sentiment analysis (which Spacy doesn't currently support)
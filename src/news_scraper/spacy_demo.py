# spacy_demo.py

'''
Goal - write a proof-of-concept script that:
- Bootstraps crypto-currency NER via IOB/BILUO (all as "MONEY")
- Trains a targeted language model (either from a pre-trained English model, or from a blank one)
  - Test this to see what performs better!
- Builds mappings between crypto-currency ticker symbols and possible aliases

Links:
- https://spacy.io/usage/training , esp the section "How do I get training data?"
- https://spacy.io/usage/linguistic-features#entity-types
- Test article: https://www.forbes.com/sites/chuckjones/2018/08/04/bitcoin-falls-under-7000/#769c8c227477
'''

# import spacy

ticker_aliases = {'Bitcoin' : 'BTC', 'bitcoin' : 'BTC', 'BTC' : 'BTC'}

article_text = '''Bitcoin rose to an intra-day high of $8,486 and close of $8,396 on July 24 after rebounding from its recent low close of $5,871 on June 28 and its intra-day low of $5,538 on July 2. However, since Tuesday Bitcoin has fallen over $1,000 to just under $7,000 on Saturday. ... Shortly after this preemptive action, unfortunately, the BTC price tumbled, causing the liquidation of the account.'''

sentences = [' '.join(s.split()) for s in article_text.split('.') if len(s) > 1]

# to adapt for a pre-trained language model, replace "[]" with the existing LM's hypothesized annotations
train_data = [(sen, []) for sen in sentences]

print('before:')
for e in train_data:
    print(e)

for i,sen in enumerate(sentences):
    # print((len(sen), sen))
    for alias in ticker_aliases:
        if alias not in sen:
            continue
        ind_start = sen.index(alias)
        ind_end = ind_start + len(alias)
        alias_annotation = (ind_start, ind_end, "MONEY")
        train_data[i][1].append(alias_annotation)

print('after:')
for e in train_data:
    print(e)
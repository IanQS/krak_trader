# Data Storage

Link to whatever file you want. 

Currently, the scraper only extracts

```
v = volume array(<today>, <last 24 hours>),
p = volume weighted average price array(<today>, <last 24 hours>),
```

from [Ticker Information](https://www.kraken.com/help/api#get-ticker-info)

This is then stored as a ```np.array``` in the specified location in the following format:

```
{'Time': array([1.52981048e+09, 1.52981049e+09, 1.52981049e+09]),

 'XETHZUSD': array([[  469.67817   ,   470.8088    ,  4459.3802086 , 35917.59522232],
       [  469.67817   ,   470.80909   ,  4459.3802086 , 35917.59522232],
       [  469.67817   ,   470.80988   ,  4459.3802086 , 35917.59522232]]),
       
 'XXLMZUSD': array([[2.00603870e-01, 1.99104590e-01, 2.93182652e+05, 1.84113851e+06],
       [2.00603870e-01, 1.99104620e-01, 2.93182652e+05, 1.84113851e+06],
       [2.00603870e-01, 1.99104540e-01, 2.93182652e+05, 1.84113851e+06]]),
       
 'XXMRZUSD': array([[ 115.71558861,  114.37026659,  210.22115849, 3973.68021051],
       [ 115.71558861,  114.37026659,  210.22115849, 3973.68021051],
       [ 115.71558861,  114.37026658,  210.22115849, 3973.68021051]]),
       
 'XXRPZUSD': array([[4.85018500e-01, 4.85766550e-01, 2.25048074e+05, 1.43422332e+06],
       [4.85018500e-01, 4.85766510e-01, 2.25048074e+05, 1.43422332e+06],
       [4.85018500e-01, 4.85766480e-01, 2.25048074e+05, 1.43422332e+06]])}

```

which you can then load easily to put into a ```pandas.DataFrame```
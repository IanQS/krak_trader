Tests the idea that we can create a lower-dimensional embedding of the orderbook

We essentially treat the orderbook as if it were an image with two channels

[Model Development Procedure](https://paper.dropbox.com/folder/show/Orderbook-Embedding-e.1gg8YzoPEhbTkrhvQwJ2zz3SRnWuDmycB9FFeXRsT5h5Q7diMUJj)

```
0 ) Test Development
    - Tests on feasibility 
    - Uses keras benchmarks  

1 ) Vanilla Autoencoder
    - Uses fully connected

2 ) Convolutional Autoencoder
    - Vanilla Version
    - L2 Loss
```
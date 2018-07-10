1) Sanity check by using AE

[Model Development Procedure](https://paper.dropbox.com/doc/Autoencoder--AHNbf_9wDe3DOehWORnUwkOjAQ-pBKUAtM8p7BFbsTf9FdQs)

2) Extend AE into Conv AE

3) LSTM architecture where the features provided are outputs fron Conv AE

[Resource](https://blog.coast.ai/continuous-video-classification-with-tensorflow-inception-and-recurrent-nets-250ba9ff6b85)




# **Refs**

1) [Conv Net features to LSTM is best for video classification](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5)

Compared against: 

    1) Classifying one frame at a time with a ConvNet
    2) Using a time-distributed ConvNet and passing the features to an RNN, in one network
    3) Using a 3D convolutional network
    4) Extracting features from each frame with a ConvNet and passing the sequence to a separate RNN
    5) Extracting features from each frame with a ConvNet and passing the sequence to a separate MLP
    
2)
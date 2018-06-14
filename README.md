# Deep learning using Keras
# Self-study 
# Under instruction of Kirill Eremenko, https://www.udemy.com/deeplearning/

# Featured project: my_rnn, a recurrent neural network predicting Google's stock price using additional indicators from Microsoft's stock prices

*Predicting Google Stock Prices:
https://github.com/tmtran11/Deep_learning/blob/master/my_rnn.py

*Using Recurrent Neural Network tuned through basics algorithm and additional indicators (using Microsoft’s stock prices).
Using pandas, numpy, plots library to process and represent data

*Data is splitted into training set and test set

*The data fed into Recurrent Neural Network to predict the stock price at a certain day is the stock prices in 120 days before

*An indicator is added, the Microsoft stock prices, which is chosen because of a strong correlation between two company.

*Pre-processing
- Using sklearn to scale the data:
- Using MinMaxScaler to scale the price to range (0,1) to reduce calulational bias

*Neural Network
- 6 layers of Long-short-term-memory cells, simulate recurrent learning, pad with dropout layer to prevent overfitting
- 1 dense layer
- The neural network is setup to optimize the min squared error

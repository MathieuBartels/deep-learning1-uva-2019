import matplotlib.pyplot as plt
data = {'RNN': [{1: 1.0}, {2: 1.0}, {3: 1.0}, {4: 1.0}, {5: 1.0}, {6: 1.0}, {7: 1.0}, {8: 1.0}, {9: 1.0}, {10: 1.0}, {11: 1.0}, {12: 1.0}, {13: 0.99925}, {14: 0.99885}, {15: 0.8814}, {16: 0.87165}, {17: 0.65145}, {18: 0.46085000000000004}, {19: 0.41974999999999996}, {20: 0.41055}, {21: 0.38079999999999997}, {22: 0.39475}, {23: 0.41655}, {24: 0.40264999999999995}], 'LSTM': [{1: 1.0}, {2: 1.0}, {3: 1.0}, {4: 1.0}, {5: 1.0}, {6: 1.0}, {7: 1.0}, {8: 1.0}, {9: 0.9798500000000001}, {10: 0.9314}, {11: 0.7463500000000001}, {12: 0.74015}, {13: 0.6813}, {14: 0.76985}, {15: 0.41135}, {16: 0.19910000000000003}, {17: 0.2206}, {18: 0.1879}, {19: 0.2824}, {20: 0.44675000000000004}, {21: 0.41035000000000005}, {22: 0.09955}, {23: 0.2572}, {24: 0.1034}]}

y_lstm = [(data['LSTM'][i][i+1]) for i in range(0,24,1)]
y_rnn = [(data['RNN'][i][i+1]) for i in range(0,24,1)]
plt.plot(list(range(0,24,1)), y_lstm)
plt.plot(list(range(0,24,1)), y_rnn)
plt.show()
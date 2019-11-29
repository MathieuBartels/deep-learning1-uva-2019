import matplotlib.pyplot as plt
import numpy as np
x = np.loadtxt('steps.npy')
accuracy = np.loadtxt('acc.npy')
errors = np.loadtxt('error.npy')


plt.figure(figsize=(9,2))
plt.title('LSTM word prediction error and accuracy')
plt.subplot(121)
plt.plot(x, accuracy)
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('step')

plt.subplot(122)

plt.plot(x, errors)
plt.title('Error')
plt.ylabel('Error')
plt.xlabel('step')

plt.show()
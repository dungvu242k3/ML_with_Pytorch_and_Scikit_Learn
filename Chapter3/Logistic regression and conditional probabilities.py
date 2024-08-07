import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z) :
    return 1.0/(1.0 + np.exp(-z))

z = np.arange(-7,7,0.1)
sigmoid_z = sigmoid(z)
plt.plot(z,sigmoid_z)
plt.axvline(0.0,color = "k")
plt.ylim(-0.1,1.1)
plt.xlabel("z")
plt.ylabel('$\sigma (z)$')
plt.yticks([0.0,0.5,1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

def loss_1(z) :
    return - np.log(sigmoid(z))

def loss_2(z) :
    return - np.log(1-sigmoid(z))

z = np.arange(-10,10,0.1)
sigma_z = sigmoid(z)

c1 = [loss_1(x) for x in z]
plt.plot(sigma_z,c1,label = 'L(w,b) if y = 1')
c2 = [loss_2(x) for x in z]
plt.plot(sigma_z,c2,linestyle = '--' ,label = 'L(w,b) if y = 0')
plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w,b)')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()
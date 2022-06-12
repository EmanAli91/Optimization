import numpy as np
import matplotlib.pyplot as plt 
import math

def func(x):
    y=3*x**3+2*x**4-3
    return y

def derv_func(x):
    derv = 9*x**2 + 8*x**3
    return derv


def gradient_descent(xi=-0.9):

    learning_rate = 0.001
    eps = 0.0001
    iter = 0
    while iter< 1000 and math.fabs(func(xi)) > eps:
        derv = derv_func(xi)
        xi -= learning_rate*derv
        iter+=1

    return xi

# Given y=3x^3+2x^4-3, try to find the x, which gives the smallest y (argmin_x (y)). 

xo = gradient_descent(xi=-0.9)
print("The local minimum occurs at", xo)

fig = plt.figure(figsize=(20,20))

## show the shape of the function y=3x^3+2x^4-3
x = np.linspace(-3,1.5,num=100,dtype=np.float32) #np.array([i for i in range(-20,20,1)]).astype(np.float32)
y = np.array([3*i**3+2*i**4-3 for i in x]).astype(np.float32)
X = np.vstack([np.reshape(x,(1,x.shape[0])),np.reshape(y,(1,y.shape[0]))])
plt.plot(X[0,:],X[1,:],'r-')

plt.plot(xo,3*xo**3+2*xo**4-3,'go') # plot the optimal point

plt.grid(True)
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.gca().set_aspect('equal',adjustable='box')
plt.show()

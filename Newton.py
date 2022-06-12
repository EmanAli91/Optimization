import numpy as np
import matplotlib.pyplot as plt
import math 

def func(x):
    y=3*x**3+2*x**4-3
    return y

def first_derv_func(x):
    derv = 9*x**2 + 8*x**3
    return derv


def second_derv_func(x):
    derv = 18*x + 24*x**2
    return derv

def Newton(xi=-0.9):
    
    eps = 0.0001
    iter = 0
    while iter< 1000 and math.fabs(func(xi)) > eps:
        first_derv = first_derv_func(xi)
        second_derv = second_derv_func(xi)
        if first_derv == 0 or second_derv == 0:
            print('Mathametical Error')
            break
        xi -= first_derv/second_derv
        iter+=1
    return xi

# Given y=3x^3+2x^4-3, try to find the x, which gives the smallest y (argmin_x (y)). 

xo = Newton(xi=-0.9)
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
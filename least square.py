import numpy as np
import matplotlib.pyplot as plt


def plot_line(l,X,marker='ro'):
    a,b,c = l
    
    x1 = X[0,0]
    y1 = (-a*x1-c)/b
    
    x2 = X[0,-1]
    y2 = (-a*x2-c)/b
    
    plt.plot([x1,x2],[y1,y2],marker)

def least_square(p):
    # p: 2xN, N is the number of points
    # y = -a/b x - c/b, or y = kx + b
    # min([[x1,1],...,[xn,1]]*[k,b]T-[[y1],...,[yn]])^2


    n = p.shape[1]
    X = p[0,:].reshape(n, 1)
    X = np.hstack((X, np.ones((n, 1))))
    Y = p[1,:].reshape(n, 1)

    ## find U, sigma, V using SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    ## find the inverse for U
    U_inverse = U.T

    ## find the inverse for V
    V = Vt.T

    ## find the inverse for Sigma    
    rank =S.shape[0]
    sigma = np.zeros((rank, rank))
    for i in range(rank):
        sigma[i, i] = S[i]
    sigma_inverse = np.linalg.inv(sigma) 

    ## Calcualte slope and intercept (beta)  
    params = V.dot(sigma_inverse).dot(U_inverse).dot(Y)

    a = -params[0]
    c = -params[1]
    b = 1
  
    l = [a, b, c]
    
    return l # l = [a,b,c], ax+by+c=0


X = np.load('least_square_X.npy')

l1 = least_square(X)

fig = plt.figure(figsize=(20,20))
plt.plot(X[0,:],X[1,:],'ro')
plot_line([2,-3,1],X,'r-') # the ground truth line
plot_line(l1,X,'g-') # plot the line fitted by the least square method
plt.grid(True)
plt.xlim((-60,60))
plt.ylim((-60,60))
plt.gca().set_aspect('equal',adjustable='box')
plt.show()

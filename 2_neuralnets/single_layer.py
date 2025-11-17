#goal for tn: currently 11/17 8:24pm is to make this single layer neural net, basically just vectorize the old version. 
#should be pretty chill
#just need to make W and B into vectors and also make the ground truth into vectors. 
#no practical use case just yet aside from having 1 neural net that can fit to 10 different functions, that's pretty cool actually. 

import matplotlib.pyplot as plt
import numpy as np


# generate ground truth and parameters.
def generate_ground_truth(n,m1,m2): 
    w_vec = []
    b_vec = []

    for i in range(n):
        w = round(np.random.rand()*m1)/m2
        b = round(np.random.rand()*m1)/m2
        w_vec.append(w)
        b_vec.append(b)
        # print(w,b)

    w_vec = np.array(w_vec)
    b_vec = np.array(b_vec)

    return w_vec,b_vec

#this p much breaks down the matrix multiplication into a for loop. need to resize so can do the native numpy operation. 
def generate_data(w,b):
    x = []
    y = []
    for i in range(10):
        x.append(i)
        y.append(w*i+b)
        print(w*i+b)

    print(x)
    print(y)
    return np.array(x),np.array(y)

def get_loss(y_hat,y):
    n = len(y_hat)
    loss = 0
    for i in range(n):
        loss += (y[i]-y_hat[i])**2
    mse = loss/n
    return mse

#derived the math myself. this was pretty fun.
def get_gradients(y_hat,y,x,w,b,alpha):
    n = len(y_hat)
    dl_dw = 0
    dl_db = 0

    for i in range(n):
        dl_dw += 2*x[i]*(y[i]-y_hat[i])
        dl_db += 2*(y[i]-y_hat[i])

    dl_dw = dl_dw/n
    dl_db = dl_db/n

    w = w + alpha*dl_dw
    b = b + alpha*dl_db
    return w,b

def main():
    alpha = 0.001
    neurons = 5

    w_ground,b_ground = generate_ground_truth(neurons,1000,100)
    x_cords, y_cords = generate_data(w_ground,b_ground)
    x_cords = x_cords.reshape(-1,1)
    print(x_cords)
    print(y_cords)

    w_vec,b_vec = generate_ground_truth(neurons,100,100)
    w_vec = w_vec.reshape(1,-1)
    print(w_vec.shape)
    print(x_cords)

    for i in range(10000):
        y_hat = np.dot(x_cords,w_vec)+b_vec
        w_vec,b_vec = get_gradients(y_hat,y_cords,x_cords,w_vec,b_vec,alpha)
        mse = get_loss(y_hat,y_cords)
            
        if i%1000 == 0:
            plt.plot(x_cords,y_hat)
            print(mse)
            print()

    #build the r
    plt.plot(x_cords,y_hat)
    plt.scatter(x_cords,y_cords)
    plt.show()

if __name__ == "__main__":
    main()
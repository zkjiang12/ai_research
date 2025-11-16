# Goal for tonight: (currently 1:44am 11/15/2025. will rest for a bit then get started.)
# Derive linear regression.
# Have it fit to a pretty simple linear function.
# Repeat for polynomial.
# Start Andrej Neural Net

import matplotlib.pyplot as plt
import numpy as np

def generate_data(w,b):
    x = []
    y = []
    for i in range(10):
        x.append(i)
        y.append(w*i+b)

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
    
    w_ground = 3
    b_ground = 4
    x_cords, y_cords = generate_data(w_ground,b_ground)

    w = np.random.rand()
    b = np.random.rand()
    for i in range(10000):
        y_hat = w*x_cords+b
        w,b = get_gradients(y_hat,y_cords,x_cords,w,b,alpha)
        mse = get_loss(y_hat,y_cords)
            
        if i%10 == 0:
            plt.plot(x_cords,y_hat)

        print(mse)

    plt.plot(x_cords,y_hat)
    plt.scatter(x_cords,y_cords)
    plt.show()

if __name__ == "__main__":
    main()
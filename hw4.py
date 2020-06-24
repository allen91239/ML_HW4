import numpy as np
import struct
import matplotlib.pyplot as plt
def uni_gaussian(mean, variance):
    s = 1
    while(s >= 1):
        u = np.random.uniform(-1,1,2)
        s = u[0]**2 + u[1]**2
        if(-2*np.log(s)/s > 0 and s < 1):
            x = u[0] * np.sqrt((-2*np.log(s)/s)) * variance + mean
            data_point = x
    data_point = np.random.normal(mean, variance)
    return data_point

def newton():
    D = np.zeros((N*2,N*2))
    w_now = np.random.uniform(size = 3)
    w_last = np.random.uniform(size = 3)
    while( (np.absolute(w_now - w_last) > 0.1).all()):
        for i in range(N*2):
            c = np.exp(-1 * np.sum(x[i] * w_now))
            D[i][i] = c / ((1 + c)**2)
        hessian = np.dot(np.dot(x.T, D), x)
        grad = np.dot(x.T, (y - 1 / (1 + np.exp(-1 * np.dot(x, w_now)))))
        if np.linalg.det(hessian) == 0:
            gradient = grad
        else:
            gradient = np.dot(np.linalg.inv(hessian), grad)
        w_last = w_now
        w_now = w_now + 0.1 * gradient
    result = 1 / (1 + np.exp(-1 * np.dot(x, w_now)))
    predict = [0 if i < 0.5 else 1 for i in result]
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(N * 2):
        if predict[i] == 0 and y[i] == 0:
            tp+=1
        elif predict[i] == 0 and y[i] == 1:
            fn+=1
        elif predict[i] == 1 and y[i] == 0:
            fp+=1
        else:
            tn+=1
        
        if(predict[i] == 0):
            plt.plot(x[i][1], x[i][2], 'ro')
        else:
            plt.plot(x[i][1], x[i][2], 'bo')
    plt.savefig("Newton")
    plt.close()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Newton:")
    print()
    print("w:")
    print(gradient[0])
    print(gradient[1])
    print(gradient[2])
    print()
    print("Confusion Matrix:")
    print("              Predict cluster 1 Predict cluster 2")
    print(f"Is cluster 1         {tp}                {fp}")
    print(f"Is cluster 2         {fn}                {tn}")
    print()
    print ("Sensitivity: ", sensitivity)
    print ("Sepcificity: ", specificity)
    print()
    print("--------------------------------------------------")

def gradient():
    gradient_now = np.random.uniform(size = 3)
    gradient_next = gradient_now + np.dot(x.T, (y - 1 / (1 + np.exp(-1 * np.dot(x, gradient_now)))))
   
    while( (np.absolute(gradient_now - gradient_next) > 0.01).all()):
        gradient_now = gradient_next
        gradient_next = gradient_next + 0.1 * np.dot(x.T, (y - 1 / (1 + np.exp(-1 * np.dot(x, gradient_now)))))
        
    gradient = gradient_next
    result = 1 / (1 + np.exp(-1 * np.dot(x, gradient)))
    predict = [0 if i < 0.5 else 1 for i in result]
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(N * 2):
        if predict[i] == 0 and y[i] == 0:
            tp+=1
        elif predict[i] == 0 and y[i] == 1:
            fn+=1
        elif predict[i] == 1 and y[i] == 0:
            fp+=1
        else:
            tn+=1
        
        if(predict[i] == 0):
            plt.plot(x[i][1], x[i][2], 'ro')
        else:
            plt.plot(x[i][1], x[i][2], 'bo')
    plt.savefig("Gradient Descent")
    plt.close()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Gradient descent:")
    print()
    print("w:")
    print(gradient[0])
    print(gradient[1])
    print(gradient[2])
    print()
    print("Confusion Matrix:")
    print("              Predict cluster 1 Predict cluster 2")
    print(f"Is cluster 1         {tp}                {fp}")
    print(f"Is cluster 2         {fn}                {tn}")
    print()
    print ("Sensitivity: ", sensitivity)
    print ("Sepcificity: ", specificity)
    print()
    print("--------------------------------------------------")
    
def read_image(image_path, label_path):
    with open(label_path, 'rb') as label:
        magic, items = struct.unpack('>II', label.read(8)) #不記前兩個
        labels = np.fromfile(label, dtype=np.uint8) 
    with open(image_path, 'rb') as image:
        magic, num, rows, cols = struct.unpack('>IIII', image.read(16)) #不記前四個
        images = np.fromfile(image, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
    
def em():
    pass
    

if __name__ == "__main__":
    global D1, D2, x, y, N
    N = int(input("Number of data points N:"))
    mx1 = int(input("X1 mean:"))
    my1 = int(input("Y1 mean:"))
    vx1 = int(input("X1 var:"))
    vy1 = int(input("Y1 var:"))

    mx2 = int(input("X2 mean:"))
    my2 = int(input("Y2 mean:"))
    vx2 = int(input("X2 var:"))
    vy2 = int(input("Y2 var:"))

    #generate data points
    
    x = np.empty((N * 2, 3))
    y = np.empty((N * 2))
    D1 = np.zeros((N, 2))
    D2 = np.zeros((N, 2))
    for i in range(N):
        D1[i][0] = uni_gaussian(mx1, vx1)
        D1[i][1] = uni_gaussian(my1, vy1)
        D2[i][0] = uni_gaussian(mx2, vx2)
        D2[i][1] = uni_gaussian(my2, vy2)
        x[i][0] = 1
        x[i][1] = D1[i][0]
        x[i][2] = D1[i][1]
        x[i+N][0] = 1
        x[i+N][1] = D2[i][0]
        x[i+N][2] = D2[i][1]
        y[i] = 0
        y[i+N] = 1
        plt.plot(D1[i][0], D1[i][1], 'ro')
        plt.plot(D2[i][0], D2[i][1], 'bo')
    plt.savefig("ground truth")
    plt.close()
    gradient()
    newton()


    train_images, train_labels = read_image('./train-images.idx3-ubyte', './train-labels.idx1-ubyte')
    train_images = train_images//128
    em()
    
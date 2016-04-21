import neural_net
import numpy as np
if __name__ == '__main__':
    #X=np.ones(10)
    X =  np.random.rand(5,5)

    for i in range(0, np.shape(X)[0]):
      for j in range(0,np.shape(X)[1]):
        if i%2 ==0:
          X[i][j] = -X[i][j]

    nn = neural_net.TwoLayerNet(4,4,4,0.01,0.95)
    print nn.Relu(X)
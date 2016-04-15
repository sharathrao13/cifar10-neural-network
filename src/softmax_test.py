import numpy as np

import neural_net

if __name__ == '__main__':
    X = np.ones(10).reshape(5,2)
    nn = neural_net.TwoLayerNet(2,2,2,0.2)
    nn.softmax(X)
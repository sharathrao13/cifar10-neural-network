import neural_net
import numpy as np
if __name__ == '__main__':
    nn = neural_net.TwoLayerNet(3072,6144,10,0.01)
    nn.train()
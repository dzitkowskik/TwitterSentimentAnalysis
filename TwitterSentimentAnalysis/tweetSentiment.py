# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import core
import numpy as np
from datasets import TweetClassificationDatasetFactory
from ai import MultiClassClassificationNeuralNetwork


def main():
    np.set_printoptions(edgeitems=20)
    dataset = TweetClassificationDatasetFactory().get_dataset()
    nn = MultiClassClassificationNeuralNetwork(3, 9, epochs=20)
    tst_data, trn_data = dataset.splitWithProportion(0.25)
    #nn.apply_custom_network([3])
    nn.run(trn_data, tst_data)
    print nn.run_with_crossvalidation(dataset)
    return

if __name__ == '__main__':
    core.initialize()
    main()
    core.terminate()
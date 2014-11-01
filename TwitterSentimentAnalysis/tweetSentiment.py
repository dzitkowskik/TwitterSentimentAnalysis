# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import core
from datasets import SimpleTweetDatasetFactory
from neuralNetworks import MultiClassClassificationNeuralNetwork


def main():
    dataset = SimpleTweetDatasetFactory().get_dataset()
    tst_data, trn_data = dataset.splitWithProportion(0.25)
    nn = MultiClassClassificationNeuralNetwork(3, 9)
    nn.run(trn_data, tst_data)
    return

if __name__ == '__main__':
    core.initialize()
    main()
    core.terminate()
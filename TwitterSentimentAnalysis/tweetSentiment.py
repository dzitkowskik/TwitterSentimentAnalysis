# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer

__author__ = 'Karol Dzitkowski'

import core
from datasets import SimpleTweetDatasetFactory
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork


class MultiClassClassificationNeuralNetwork(object):
    def __init__(self, inp_cnt, out_cnt, hid_cnt=10, epochs=100):
        self.hid_cnt = hid_cnt
        self.out_cnt = out_cnt
        self.inp_cnt = inp_cnt
        self.epochs = epochs
        self.network = self.__build_default_network()

    # def build_custom_network(self, hidden_counts):
    #     network = FeedForwardNetwork()
    #     in_layer = LinearLayer(self.inp_cnt)
    #     hidden_layer = SigmoidLayer(hidden_count)
    #     out_layer = SoftmaxLayer(self.out_cnt)
    #     network.addInputModule(in_layer)
    #     network.addModule(hidden_layer)
    #     network.addOutputModule(out_layer)
    #     in_to_hidden = FullConnection(in_layer, hidden_layer)
    #     hidden_to_out = FullConnection(hidden_layer, out_layer)
    #     network.addConnection(in_to_hidden)
    #     network.addConnection(hidden_to_out)
    #     network.sortModules()
    #     return network

    def __build_default_network(self):
        return buildNetwork(self.inp_cnt, self.hid_cnt, self.out_cnt, outclass=SoftmaxLayer, bias=True)

    def run(self, ds_train, ds_test):
        trainer = BackpropTrainer(
            self.network,
            dataset=ds_train,
            momentum=0.1,
            verbose=True,
            weightdecay=0.01)

        trainer.trainEpochs(self.epochs)
        tstresult = percentError(
            trainer.testOnClassData(dataset=ds_test),
            ds_test['class'])

        print "Multi class neural network test error: %5.2f%%" % tstresult
        return tstresult

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)


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
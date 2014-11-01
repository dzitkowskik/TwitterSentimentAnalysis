__author__ = 'ghash'

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
from abc import ABCMeta, abstractmethod


class NeuralNetwork(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, ds_train, ds_test):
        pass

    @abstractmethod
    def __call__(self, ds_train, ds_test):
        pass


class MultiClassClassificationNeuralNetwork(NeuralNetwork):
    def __init__(self, inp_cnt, out_cnt, hid_cnt=10, epochs=100):
        self.hid_cnt = hid_cnt
        self.out_cnt = out_cnt
        self.inp_cnt = inp_cnt
        self.epochs = epochs
        self.network = self.__build_default_network()

    def apply_custom_network(self, hidden_counts):
        network = FeedForwardNetwork()
        in_layer = LinearLayer(self.inp_cnt)
        network.addInputModule(in_layer)
        out_layer = SoftmaxLayer(self.out_cnt)
        network.addOutputModule(out_layer)

        hidden_layer = SigmoidLayer(hidden_counts[0])
        network.addModule(hidden_layer)
        in_to_hidden = FullConnection(in_layer, hidden_layer)
        network.addConnection(in_to_hidden)

        for i in range(1, len(hidden_counts)):
            last_hidden_layer = hidden_layer
            hidden_layer = SigmoidLayer(hidden_counts[i])
            network.addModule(hidden_layer)
            hidden_to_hidden = FullConnection(last_hidden_layer, hidden_layer)
            network.addConnection(hidden_to_hidden)

        hidden_to_out = FullConnection(hidden_layer, out_layer)
        network.addConnection(hidden_to_out)
        network.sortModules()
        self.network = network
        return self

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

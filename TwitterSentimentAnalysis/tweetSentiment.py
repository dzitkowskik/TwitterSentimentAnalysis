__author__ = 'Ghash'

import inject
import core
from config import Config
from pymongo import MongoClient
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError


def build_full_connected_network(input_count, hidden_count, output_count):
    network = FeedForwardNetwork()
    in_layer = LinearLayer(input_count)
    hidden_layer = SigmoidLayer(hidden_count)
    out_layer = LinearLayer(output_count)
    network.addInputModule(in_layer)
    network.addModule(hidden_layer)
    network.addOutputModule(out_layer)
    in_to_hidden = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)
    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)
    network.sortModules()
    return network


class TweetDatasetFactory(object):
    labels = [
        'Negative (-4)',
        'Negative (-3)',
        'Negative (-2)',
        'Negative (-1)',
        'Neutral',
        'Positive (+1)',
        'Positive (+2)',
        'Positive (+3)',
        'Positive (+4)']

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    def __create_classification_dataset(self):
        return ClassificationDataSet(3, 1, 9, class_labels=self.labels)

    @staticmethod
    def __get_input_from_record(record):
        favorite_count = record['data']['favorite_count']
        followers_count = record['data']['user']['followers_count']
        retweet_count = record['data']['retweet_count']
        return favorite_count, followers_count, retweet_count

    @staticmethod
    def __get_output_from_record(record):
        # word_sentiment is stored as float between -5 and 5
        scale = 4.0 / 5.0
        word_grade = abs(record['word_sentiment']) * scale  # between 0 and 4
        manual_grade = 0
        if record['manual_grade'] == 'positive':
            manual_grade = 1
        elif record['manual_grade'] == 'negative':
            manual_grade = -1

        return round(word_grade * manual_grade)

    def get_dataset(self):
        ds = self.__create_classification_dataset()
        for record in self.db.train_data.find({"isActive": True}):
            ds.addSample(self.__get_input_from_record(record), self.__get_output_from_record(record))
        return ds


def main():
    network = build_full_connected_network(3, 3, 1)
    dataset = TweetDatasetFactory().get_dataset()
    tst_data, trn_data = dataset.splitWithProportion(0.25)
    trainer = BackpropTrainer(network, dataset=trn_data, momentum=0.1, verbose=True, weightdecay=0.01)
    for i in range(20):
        trainer.trainEpochs(10)
        trn_result = percentError(trainer.testOnClassData(), trn_data['class'])
        tst_result = percentError(trainer.testOnClassData(dataset=tst_data), tst_data['class'])

        print "epoch: %4d" % trainer.totalepochs, "  train error: %2.5f%%" % trn_result, \
            "  test error: %2.5f%%" % tst_result
    return


if __name__ == '__main__':
    core.initialize()
    main()
    core.terminate()
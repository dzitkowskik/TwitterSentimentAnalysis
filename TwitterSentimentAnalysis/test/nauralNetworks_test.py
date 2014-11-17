import uuid
import unittest
from TwitterSentimentAnalysis import ai, core, downloaders, datasets
import numpy as np
import os


class NeuralNetworksTweetsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(__file__)
        configuration_file_path = os.path.join(base_dir, 'test_configuration.cfg')
        core.initialize(configuration_file_path)

    @classmethod
    def tearDownClass(cls):
        core.terminate()

    def setUp(self):
        self.tweet_downloader = downloaders.TweetDownloader()
        self.tweetclassificationdataset = datasets.TweetClassificationDatasetFactory()
        self.test_db = self.tweet_downloader.db
        self.test_table_name = "tweet_download_" + uuid.uuid4().hex + "_test"

    def tearDown(self):
        self.test_db.drop_collection(self.test_table_name)

    def test_multi_class_classification_neural_network(self):
        neural_network = ai.MultiClassClassificationNeuralNetwork(3, 9)
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        self.assertIsNotNone(neural_network.network)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = neural_network.run(ds_train, ds_test)
        actual = neural_network.network.activateOnDataset(ds_test)
        expected = ds_test['class']
        expected_error = (np.argmax(actual, 1) != expected.T).mean(dtype=float)
        self.assertEqual(result, expected_error)

    def test_naive_bayes_classifier(self):
        # TODO: finish
        classifier = ai.NaiveBayesClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        self.assertIsNotNone(classifier.classifier)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = classifier.run(ds_train, ds_test)
        res = classifier.classify_many(ds_test['input'])
        print res
        print ds_test['target']
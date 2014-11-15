import uuid
import unittest
from TwitterSentimentAnalysis import neuralNetworks, core, downloaders, datasets
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from config import Config
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

    def test_MultiClassClassificationNeuralNetwork(self):
        neural_network = neuralNetworks.MultiClassClassificationNeuralNetwork(3, 1)
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag = "erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)

        self.assertIsNotNone(neural_network.network)

        ds_train, ds_test = ds.splitWithProportion(0.75)

        result = neural_network.run(ds_train, ds_test)

        tstresult = percentError(
            neural_network.network.activateOnDataset(ds_test),
            ds_test['class'])


        self.assertEqual(result, tstresult)




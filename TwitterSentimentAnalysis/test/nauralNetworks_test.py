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
        self.tweetregressiondataset = datasets.TweetRegressionDatasetFactory()
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
        self.assertEqual(result/100, expected_error)

    def test_simple_regression_neural_network(self):
        neural_network = ai.SimpleRegressionNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetregressiondataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = neural_network.run(ds_train, ds_test)

        actual = neural_network.network.Trainer.module.activateOnDataset(ds_test)
        expected = ds_test['target']

        tot = 0
        for i, x in enumerate(expected):
            if x == actual[i]:
                tot = tot + 1

        expected_error = float(tot)/float(len(expected))

        self.assertEqual(100-result[0], expected_error)

    def test_simple_classification_neural_network(self):
        neural_network = ai.SimpleClassificationNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = neural_network.run(ds_train, ds_test)

        actual = neural_network.network.Trainer.module.activateOnDataset(ds_test)
        expected = ds_test['target']

        print actual
        print expected

        expected_error = (np.argmax(actual, 1) != expected.T).mean(dtype=float)

        self.assertEqual(result[0]/100, expected_error)

    def test_naive_bayes_classifier(self):
        classifier = ai.NaiveBayesClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = classifier.run(ds_train, ds_test)
        self.assertIsNotNone(classifier.classifier)

        test_ds = []
        for i, k in enumerate(ds_test['input']):
            features = {}
            features['first'] = ds_test['input'][i][0]
            features['second'] = ds_test['input'][i][1]
            features['third'] = ds_test['input'][i][2]
            test_ds.append(features)

        res = []
        for i, test_rec in enumerate(test_ds):
            res.append(classifier.classifier.classify(test_rec))

        target = []
        for x in ds_test['target']:
            if x[0] > 0:
                target.append(1)
            elif x[0] == 0:
                target.append(0)
            else:
                target.append(-1)

        tot = 0
        for i, x in enumerate(target):
            if x == res[i]:
                tot = tot + 1

        expected_error = float(tot)/float(len(target))

        self.assertEqual(result, expected_error)

    def test_max_ent_classifier(self):
        classifier = ai.MaxEntropyClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = classifier.run(ds_train, ds_test)
        self.assertIsNotNone(classifier.classifier)

        test_ds = []
        for i, k in enumerate(ds_test['input']):
            features = {}
            features['first'] = ds_test['input'][i][0]
            features['second'] = ds_test['input'][i][1]
            features['third'] = ds_test['input'][i][2]
            test_ds.append(features)

        res = []
        for i, test_rec in enumerate(test_ds):
            res.append(classifier.classifier.classify(test_rec))

        target = []
        for x in ds_test['target']:
            if x[0] > 0:
                target.append(1)
            elif x[0] == 0:
                target.append(0)
            else:
                target.append(-1)

        tot = 0
        for i, x in enumerate(target):
            if x == res[i]:
                tot = tot + 1

        expected_error = float(tot)/float(len(target))

        self.assertEqual(result, expected_error)

    def test_linear_regression(self):
        model = ai.LinearRegression()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetregressiondataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = model.run(ds_train, ds_test)

        X_test = ds_test['input']
        y_test = ds_test['target']

        actual = model.regression.predict(X_test) # y_pred
        expected = ds_test['target'] # y_true

        expected_error = 1-((expected - actual) ** 2).sum()/((expected - expected.mean()) ** 2).sum()

        self.assertEqual(result, expected_error)



def main():
    test = NeuralNetworksTweetsTestCase()
    test.test_simple_classification_neural_network()

if __name__ == "__main__":
    main()





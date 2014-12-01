import uuid
import unittest
from TwitterSentimentAnalysis import ai, core, downloaders, datasets
import numpy as np
import os
from sklearn.metrics import mean_squared_error


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
        self.tweet_regression_dataset = datasets.TweetRegressionDatasetFactory()
        self.test_db = self.tweet_downloader.db
        self.test_table_name = "tweet_download_" + uuid.uuid4().hex + "_test"
        self.file_path = ""

    def tearDown(self):
        self.test_db.drop_collection(self.test_table_name)
        if self.file_path != "":
            os.remove(self.file_path)

    def test_multi_class_classification_neural_network(self):
        neural_network = ai.MultiClassClassificationNeuralNetwork(4, 9)
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        self.assertIsNotNone(neural_network.network)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = neural_network.run(ds_train, ds_test)
        actual = neural_network.network.activateOnDataset(ds_test)
        expected = ds_test['class']
        expected_error = np.mean((np.argmax(actual, 1) != expected.T), dtype=float)
        self.assertEqual(result/100, expected_error)

    def test_simple_regression_neural_network(self):
        neural_network = ai.SimpleRegressionNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweet_regression_dataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = neural_network.run(ds_train, ds_test)
        actual = neural_network.network.Trainer.module.activateOnDataset(ds_test)
        error = mean_squared_error(actual, ds_test['target'])
        self.assertEqual(result, error)

    def test_simple_classification_neural_network(self):
        neural_network = ai.SimpleClassificationNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = neural_network.run(ds_train, ds_test)
        actual = neural_network.network.Trainer.module.activateOnDataset(ds_test)
        expected = ds_test['target']
        expected_error = np.mean((np.argmax(actual, 1) != expected.T), dtype=float)
        self.assertEqual(result/100, expected_error)

    def test_naive_bayes_classifier(self):
        classifier = ai.NaiveBayesClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = classifier.run(ds_train, ds_test)
        self.assertIsNotNone(classifier.classifier)
        test_ds = []
        for i, k in enumerate(ds_test['input']):
            features = {
                'first': ds_test['input'][i][0],
                'second': ds_test['input'][i][1],
                'third': ds_test['input'][i][2],
                'fourth': ds_test['input'][i][3]}
            test_ds.append(features)
        res = []
        for i, test_rec in enumerate(test_ds):
            res.append(classifier.classifier.classify(test_rec))
        tot = 0
        for i, x in enumerate(ds_test['target']):
            if x == res[i]:
                tot += 1
        expected_error = 1-float(tot)/float(len(ds_test['target']))
        self.assertAlmostEqual(result/100, expected_error)

    def test_max_ent_classifier(self):
        classifier = ai.MaxEntropyClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = classifier.run(ds_train, ds_test)
        self.assertIsNotNone(classifier.classifier)
        test_ds = []
        for i, k in enumerate(ds_test['input']):
            features = {
                'first': ds_test['input'][i][0],
                'second': ds_test['input'][i][1],
                'third': ds_test['input'][i][2],
                'fourth': ds_test['input'][i][3]}
            test_ds.append(features)
        res = []
        for i, test_rec in enumerate(test_ds):
            res.append(classifier.classifier.classify(test_rec))
        tot = 0
        for i, x in enumerate(ds_test['target']):
            if x == res[i]:
                tot += 1
        expected_error = 1-float(tot)/float(len(ds_test['target']))
        self.assertAlmostEqual(result/100, expected_error)

    def test_linear_regression(self):
        model = ai.LinearRegression()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweet_regression_dataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        result = model.run(ds_train, ds_test)
        x_test = ds_test['input']
        actual = model.regression.predict(x_test)  # y_pred
        error = mean_squared_error(ds_test['target'], actual)
        self.assertEqual(result, error)

    def test_save_multiclassclassification(self):
        network_before = ai.MultiClassClassificationNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        res_before = network_before.run(ds_train, ds_test)
        base_dir = os.path.dirname(__file__)
        network_name = 'network' + uuid.uuid4().hex + '_test'
        self.file_path = os.path.join(base_dir, network_name)
        network_before.save(self.file_path)
        network_after = ai.MultiClassClassificationNeuralNetwork()
        network_after.load(network_name)
        res_after = network_after.test(ds_test)
        self.assertEqual(res_before, res_after)

    def test_save_simpleregression(self):
        network_before = ai.SimpleRegressionNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweet_regression_dataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        res_before = network_before.run(ds_train, ds_test)
        base_dir = os.path.dirname(__file__)
        network_name = 'network'+uuid.uuid4().hex+'_test'
        self.file_path = os.path.join(base_dir, network_name)
        network_before.save(self.file_path)
        network_after = ai.SimpleRegressionNeuralNetwork()
        network_after.load(network_name)
        res_after = network_after.test(ds_test)
        self.assertEqual(res_before, res_after)

    def test_save_simpleclassification(self):
        network_before = ai.SimpleClassificationNeuralNetwork()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        res_before = network_before.run(ds_train, ds_test)
        base_dir = os.path.dirname(__file__)
        network_name = 'network'+uuid.uuid4().hex+'_test'
        self.file_path = os.path.join(base_dir, network_name)
        network_before.save(self.file_path)
        network_after = ai.SimpleClassificationNeuralNetwork()
        network_after.load(network_name)
        res_after = network_after.test(ds_test)
        self.assertEqual(res_before, res_after)

    def test_save_naivebayes(self):
        classifier_before = ai.NaiveBayesClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        res_before = classifier_before.run(ds_train, ds_test)
        base_dir = os.path.dirname(__file__)
        network_name = 'network'+uuid.uuid4().hex+'_test'
        self.file_path = os.path.join(base_dir, network_name)
        classifier_before.save(self.file_path)
        classifier_after = ai.NaiveBayesClassifier()
        classifier_after.load(network_name)
        res_after = classifier_after.test(ds_test)
        self.assertEqual(res_before, res_after)

    def test_save_maxentropy(self):
        classifier_before = ai.MaxEntropyClassifier()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        res_before = classifier_before.run(ds_train, ds_test)
        base_dir = os.path.dirname(__file__)
        classifier_name = 'network'+uuid.uuid4().hex+'_test'
        self.file_path = os.path.join(base_dir, classifier_name)
        classifier_before.save(self.file_path)
        classifier_after = ai.MaxEntropyClassifier()
        classifier_after.load(classifier_name)
        res_after = classifier_after.test(ds_test)
        self.assertEqual(res_before, res_after)

    def test_save_linearregression(self):
        regression_before = ai.LinearRegression()
        self.tweet_downloader.download_tweets_using_query("erasmus", 100, self.test_table_name, tag="erasmus")
        ds = self.tweet_regression_dataset.get_dataset(self.test_table_name)
        ds_train, ds_test = ds.splitWithProportion(0.75)
        res_before = regression_before.run(ds_train, ds_test)
        base_dir = os.path.dirname(__file__)
        regression_name = 'network'+uuid.uuid4().hex+'_test'
        self.file_path = os.path.join(base_dir, regression_name)
        regression_before.save(self.file_path)
        regression_after = ai.LinearRegression()
        regression_after.load(regression_name)
        res_after = regression_after.test(ds_test)
        self.assertEqual(res_before, res_after)
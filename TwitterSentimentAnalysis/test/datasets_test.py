import uuid
import unittest
from TwitterSentimentAnalysis import core, downloaders, datasets
import os
from datetime import datetime


class DataSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(__file__)
        configuration_file_path = os.path.join(base_dir, 'test_configuration.cfg')
        core.initialize(configuration_file_path)

    @classmethod
    def tearDownClass(cls):
        core.terminate()

    def setUp(self):
        self.tweetclassificationdataset = datasets.TweetClassificationDatasetFactory()
        self.tweetregressiondataset = datasets.TweetRegressionDatasetFactory()
        self.tweet_downloader = downloaders.TweetDownloader()
        self.test_db = self.tweet_downloader.db
        self.test_table_name = "tweet_download_" + uuid.uuid4().hex + "_test"

    def test_get_dataset_classification(self):
        expected = 100
        self.tweet_downloader.download_tweets_using_query("#erasmus", expected, self.test_table_name, tag="erasmus")
        ds = self.tweetclassificationdataset.get_dataset(self.test_table_name)
        table = self.test_db[self.test_table_name]
        self.assertEqual(ds['input'][0][0], table.find_one()["data"]["favorite_count"])
        self.assertEqual(ds['input'][0][1], table.find_one()["data"]['user']['followers_count'])
        self.assertEqual(ds['input'][0][2], table.find_one()['retweet_count'])
        self.assertEqual(expected, table.count())

    def test_get_dataset_regression(self):
        expected = 100
        self.tweet_downloader.download_tweets_using_query("#erasmus", expected, self.test_table_name, tag="erasmus")
        ds = self.tweetregressiondataset.get_dataset(self.test_table_name)
        table = self.test_db[self.test_table_name]
        self.assertEqual(ds['input'][0][0], table.find_one()["data"]["favorite_count"])
        self.assertEqual(ds['input'][0][1], table.find_one()["data"]['user']['followers_count'])
        self.assertEqual(ds['input'][0][2], table.find_one()['word_sentiment'])
        self.assertEqual(ds['input'][0][3], (datetime.now() - datetime.strptime(table.find_one()['data']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')).days)
        self.assertEqual(expected, table.count())
# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import os
import inject
import unittest
from TwitterSentimentAnalysis import core
from config import Config
from pymongo import MongoClient
from tweepy import API


class CoreTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(__file__)
        configuration_file_path = os.path.join(base_dir, 'test_configuration.cfg')
        core.initialize(configuration_file_path)

    @classmethod
    def tearDownClass(cls):
        core.terminate()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_core_initialize_inject_config(self):
        config = inject.instance(Config)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, Config)

    def test_core_initialize_inject_db(self):
        client = inject.instance(MongoClient)
        self.assertIsNotNone(client)
        self.assertIsInstance(client, MongoClient)

    def test_core_initialize_tweepy_api(self):
        api = inject.instance(API)
        self.assertIsNotNone(api)
        self.assertIsInstance(api, API)

    def test_convert_rel_to_absolute_simple(self):
        path = core.convert_rel_to_absolute('test.test')
        expected = core.base_dir + '/test.test'
        self.assertEqual(path, expected)

    def test_convert_rel_to_absolute_relative(self):
        path = core.convert_rel_to_absolute('../test.test')
        expected = os.path.dirname(core.base_dir) + '/test.test'
        self.assertEqual(path, expected)
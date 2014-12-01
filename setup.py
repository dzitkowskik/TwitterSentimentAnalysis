#!/usr/bin/env python

# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

from config import Config
from TwitterSentimentAnalysis import core, downloaders
from setuptools import setup
from setuptools.command.install import install
from distutils.command.install import install as _install


class InstallCustom(install):
    # inject your own code into this func as you see fit
    def run(self):
        core.initialize()
        cfg = Config(core.configuration_file_path)
        table = cfg.train_tweets_table
        limit = cfg.pred_tweet_limit
        downloaders.TweetDownloader().download_tweets_from_file(table_name=table, limit=limit)
        core.terminate()
        return _install.run(self)


setup(name='TwitterSentimentAnalysis',
      version='1.0.0',
      description='Twitter Sentiment Analysis',
      author='Karol Dzitkowski & Matthias Baetens',
      author_email='k.dzitkowski@gmail.com',
      keywords='twitter sentiment analysis retweet',
      packages=['TwitterSentimentAnalysis'],
      cmdclass={'install': InstallCustom},)

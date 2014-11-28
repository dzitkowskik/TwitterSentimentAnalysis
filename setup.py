#!/usr/bin/env python

# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

from config import Config
from TwitterSentimentAnalysis import core, downloaders

core.initialize()
print "Downloading tweets...\n"

cfg = Config(core.configuration_file_path)
downloaders.TweetDownloader().download_tweets_from_file(table_name=cfg.train_tweets_table)

print '\nDownloading tweets done!\n'
core.terminate()
#!/usr/bin/env python

# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import sys
from config import Config
from TwitterSentimentAnalysis import core, downloaders
from setuptools import setup
from setuptools.command.install import install
from distutils.command.install import install as _install


class InstallCustom(install):
    # inject your own code into this func as you see fit
    def run(self):
        ret = None
        if self.old_and_unmanageable or self.single_version_externally_managed:
            ret = _install.run(self)
        else:
            caller = sys._getframe(2)
            caller_module = caller.f_globals.get('__name__','')
            caller_name = caller.f_code.co_name

            if caller_module != 'distutils.dist' or caller_name!='run_commands':
                _install.run(self)
            else:
                self.do_egg_install()

                core.initialize()

        table = Config(core.configuration_file_path).train_tweets_table
        downloaders.TweetDownloader().download_tweets_from_file(table_name=table)
        core.terminate()
        return ret


setup(name='TwitterSentimentAnalysis',
      version='1.0.0',
      description='Twitter Sentiment Analysis',
      author='Karol Dzitkowski & Matthias Baetens',
      author_email='k.dzitkowski@gmail.com',
      keywords='twitter sentiment analysis retweet',
      packages=['TwitterSentimentAnalysis', 'django_chartit_1_7'],
      cmdclass={'install': InstallCustom},)
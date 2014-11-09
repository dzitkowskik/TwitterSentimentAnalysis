# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import inject
import config
import pymongo
import tweepy
import json
import sys


reload(sys)
sys.setdefaultencoding('utf-8')
configuration_file_path = 'configuration.cfg'


# noinspection PyDecorator
# this decorator @classmethod must be here
@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status


def get_tweepy_api(cfg, json_status_parse=True):

    if json_status_parse:
        tweepy.models.Status.first_parse = tweepy.models.Status.parse
        tweepy.models.Status.parse = parse

    # == OAuth Authentication ==
    auth = tweepy.OAuthHandler(cfg.consumer_key, cfg.consumer_secret)
    auth.secure = True
    auth.set_access_token(cfg.access_token, cfg.access_token_secret)

    # Get tweepy api
    return tweepy.API(auth)


def __main_config(binder):
    cfg = config.Config(file(configuration_file_path))
    db_client = pymongo.MongoClient(cfg.db_host, int(cfg.db_port))
    tweepy_api = get_tweepy_api(cfg)

    # bind class implementations
    binder.bind(config.Config, cfg)
    binder.bind(pymongo.MongoClient, db_client)
    binder.bind(tweepy.API, tweepy_api)


def initialize():
    inject.configure(__main_config)


def terminate():
    inject.instance(pymongo.MongoClient).close()


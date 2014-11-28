# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import os
import inject
import config
import pymongo
import tweepy
import json
from tweepy import models

base_dir = os.path.dirname(__file__)
configuration_file_path = os.path.join(base_dir, 'configuration.cfg')
__core_initialized = False


def get_tweepy_api(cfg):

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


def __tweepy_parse_json():
    # settings for tweepy api
    # downloading also whole tweet and parsing as json
    # noinspection PyDecorator
    @classmethod
    def json_parse(cls, api, raw):
        status = cls.first_parse(api, raw)
        setattr(status, 'json', json.dumps(raw))
        return status
    models.Status.first_parse = models.Status.parse
    models.Status.parse = json_parse


def initialize(conf_file_name=None):
    """
    Main initialization function
    :param conf_file_name:
    :return:
    """
    global __core_initialized
    if __core_initialized:
        return
    global configuration_file_path
    if conf_file_name is not None:
        configuration_file_path = os.path.join(base_dir, conf_file_name)
    inject.configure(__main_config)
    __tweepy_parse_json()
    __core_initialized = True


def terminate():
    global __core_initialized
    if __core_initialized:
        inject.instance(pymongo.MongoClient).close()


def convert_rel_to_absolute(rel):
    return os.path.normpath(os.path.join(base_dir, rel))


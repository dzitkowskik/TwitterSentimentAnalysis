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

# -*- coding: utf-8 -*-
"""A core module of TwitterSentimentAnalysis package

This module is used for an initialization of an API (dependency injection,
configuration data and tweepy API access)

Attributes:
  base_dir (string): a base package directory used for calculating file paths
  configuration_file_path (string): a path to a main configuration file
  __core_initialized (boolean): indicates if a core has been initialized
"""

base_dir = os.path.dirname(__file__)
configuration_file_path = os.path.join(base_dir, 'configuration.cfg')
__core_initialized = False


def get_tweepy_api(cfg):
    """
    Provide OAuth authentication to twitter API and initializes tweepy API
    :param cfg: Configuration of TwitterSentimentAnalysis
    :return: initialized tweepy API
    """
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
    """This function initializes whole TwitterSentimentAnalysis package

    It must be called before any use of TwitterSentimentAnalysis package

    Args:
      conf_file_name (string): A path to non default configuration file
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
    """This function terminates whole TwitterSentimentAnalysis package

    It must be called after the end of usage of TwitterSentimentAnalysis package
    It closes a connection to DB if core was initialized

    """
    global __core_initialized
    if __core_initialized:
        inject.instance(pymongo.MongoClient).close()


def convert_rel_to_absolute(rel):
    """
    Converts a path relative to base package directory to an absolute system path
    :param rel: path relative to base package directory
    :return: absolute system path
    """
    return os.path.normpath(os.path.join(base_dir, rel))


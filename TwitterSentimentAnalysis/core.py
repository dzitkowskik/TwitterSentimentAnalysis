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
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
  Examples can be given using either the ``Example`` or ``Examples``
  sections. Sections support any reStructuredText formatting, including
  literal blocks::

      $ python example_google.py

Section breaks are created by simply resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
  module_level_variable (int): Module level variables may be documented in
    either the ``Attributes`` section of the module docstring, or in an
    inline docstring immediately following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.

.. _Google Python Style Guide:
   http://google-styleguide.googlecode.com/svn/trunk/pyguide.html

"""

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
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If the parameter itself is optional, it should be noted by adding
    ", optional" to the type. If \*args or \*\*kwargs are accepted, they
    should be listed as \*args and \*\*kwargs.

    The format for a parameter is::

        name (type): description
          The description may span multiple lines. Following
          lines should be indented.

          Multiple paragraphs are supported in parameter
          descriptions.

    Args:
      param1 (int): The first parameter.
      param2 (str, optional): The second parameter. Defaults to None.
        Second line of description should be indented.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      bool: True if successful, False otherwise.

      The return type is optional and may be specified at the beginning of
      the ``Returns`` section followed by a colon.

      The ``Returns`` section may span multiple lines and paragraphs.
      Following lines should be indented to match the first line.

      The ``Returns`` section supports any reStructuredText formatting,
      including literal blocks::

          {
              'param1': param1,
              'param2': param2
          }

    Raises:
      AttributeError: The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
      ValueError: If `param2` is equal to `param1`.

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


__author__ = 'Ghash'

import inject
import config
import pymongo

def __main_config(binder):
    cfg = config.Config(file('configuration.cfg'))
    db_client = pymongo.MongoClient(cfg.db_host, int(cfg.db_port))

    binder.bind(config.Config, cfg)
    binder.bind(pymongo.MongoClient, db_client)


def configure():
    inject.configure(__main_config)


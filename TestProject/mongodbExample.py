__author__ = 'ghash'

from pymongo import MongoClient

client = MongoClient()
db = client.mytestdb
collection = db.testData

print "collection names: ", db.collection_names()

client.close()



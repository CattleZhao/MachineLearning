# -*- coding: utf-8 -*-

import pymongo

conn = pymongo.MongoClient();
db = conn.test;
test = db['test'];
print(test.find_one());
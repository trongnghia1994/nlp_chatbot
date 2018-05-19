# -*- coding: utf-8 -*-
import json, codecs


class DatasetApis:
    def __init__(self):
        pass

    @staticmethod
    def read_dataset(fp):
        return json.load(codecs.open(fp, 'r', 'utf-8-sig'))

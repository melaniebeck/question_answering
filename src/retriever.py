#######################################
#  Class and utils for the Retriever  #
#######################################

# What does the RETRIEVER need? 
# a method for instantiation
# pipeline to connect to a running instance
# query processing
# document fetching
# document processing

# What classes does the RETRIEVER need to interact with? 
# EXAMPLE class: if this is an evaluation setting, the example class will have the document title 
# PIPELINE class: the thing that connects the retriever to the reader
import re
import os
import json
import pickle as pkl
import logging
import time
from tqdm import tqdm
from elasticsearch import Elasticsearch

def load_corpus(filename):
    if os.path.exists(filename):
        corpus = pkl.load(open(filename, 'rb'))
        return corpus
    print(r"{filename} does not exist!")
    return

class Retriever():

    def __init__(
        self, 
        index_name='demo_index',
        settings={
            "mappings": {
                "dynamic": "strict",        
                "properties": {
                    "document_title": {"type": "text"},
                    "document_url": {"type": "text"},
                    "document_text_clean": {"type": "text"}
                    }
                }
            },
        corpus_filename = None
        ):
        self.index_name = index_name
        self.settings = settings
        
        if corpus_filename:
            evidence_corpus = load_corpus(corpus_filename)
        else:
            evidence_corpus = load_corpus('evidence_corpus_mini.pkl')

        self.es = self.connect_es()
        self.create_es_index()
        self.load_es_index(evidence_corpus)

    def connect_es(self, host='localhost', port=9200):
        '''
        Instantiate and return a Python ElasticSearch object

        Args:
            host (str)
            port (int)
        
        Returns:
            es (elasticsearch.client.Elasticsearch)

        '''
        
        config = {'host':host, 'port':port}

        try:
            es = Elasticsearch([config])

        except Exception as e:
            logging.error('Couldnt connect to ES server', exc_info=e)

        return es

    def create_es_index(self):
        '''
        Create an ElasticSearch index

        Args:
            es_obj (elasticsearch.client.Elasticsearch)
            settings (dict)
            index_name (str)

        '''

        self.es.indices.create(index=self.index_name, body=self.settings, ignore=400)

        logging.info('Index created successfully!')

        return

    def load_es_index(self, evidence_corpus):
        '''
        Loads records into an existing ElasticSearch index

        Args:
            es_obj (elasticsearch.client.Elasticsearch)
            index_name (str)
            evidence_corpus (list) - list of dicts containing data records

        '''

        for i, rec in enumerate(tqdm(evidence_corpus)):
        
            try:
                index_status = self.es.index(index=self.index_name, id=i, body=rec)

            except Exception as e:
                logging.error(f'Error loading doc with index {i}', exc_info=e)
        
        time.sleep(10)
        n_records = self.es.count(index=self.index_name)['count']
        logging.info(f'Succesfully loaded {n_records} into {self.index_name}')

        return

    def run_question_query(self, question_text, n_results=5):
        '''

        '''
        # construct query
        query = {
                'query': {
                    'query_string': {
                        'query': re.sub('[^A-Za-z0-9]+', ' ', question_text),
                        'default_field': 'document_text_clean'
                        }
                    }
                }

        # execute query
        res = self.es.search(index=self.index_name, body=query, size=n_results)

        #TODO run some checks / cleaning on the results - return something easier to parse

        return res
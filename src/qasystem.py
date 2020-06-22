#############################################################################
#  Class that combines the Reader and the Retriever into a Pipeline system  #
#############################################################################
import os
import pickle
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import squad_evaluate
from reader import Reader
from retriever import Retriever

class QASystem:
    # A qa pipeline needs a reader and a retriever
    def __init__(
        self,
        reader = None,
        retriever = None,
        topk = 1,
        corpus_filename = None,
        evidence_corpus = None
        ):
        # load the default QAsystem
        self.topk = topk
        self.reader = Reader()
        self.retriever = Retriever(
            corpus_filename=corpus_filename,
            evidence_corpus=evidence_corpus
            )

    def query(self, question):
        retriever_results = self.retriever.run_question_query(question, n_results=self.topk)

        passages = retriever_results['hits']['hits']   #['_source']['document_text_clean']
        docs = []
        for passage in passages:
            doc = {
                'id': passage['_id'],
                'score': passage['_score'],
                'text': passage['_source']['document_text'], #_clean
                'title': passage['_source']['document_title'],
                #'url': passage['_source']['document_url']
            }
            docs.append(doc)
        self.passages = docs
        answers = self.reader.predict(question, self.passages)
        return answers

    def evaluate(self, examples, output_path='./', filename="qasystem_predictions_.pkl"):
        # evaluate the end-to-end qa system

        # Process:
        # for each example (for now we will rely on these being squad examples)
        #   pass to the retriever and get top n passages
        #   pass these passages to the reader and get top n answers
        #   save the predictions  
        #
        # run squad_evaluate basically -- feed exmaples and predictions
        outfile = output_path+filename

        if os.path.exists(outfile):
            predictions = pickle.load(open(outfile, "rb"))
        else:
            predictions = {}
            meta_predictions = {}
            for example in tqdm(examples):
                reader_results = self.query(example.question_text)
                answers = reader_results['answers']
                predictions[example.qas_id] = answers[0]['answer_text']

                # for debugging/explainability - save the full answer 
                # (not just text answer from top hit)
                meta_predictions[example.qas_id] = answers

            pickle.dump(predictions, open(outfile, "wb"))

            meta_outfile = os.path.splitext(outfile)[0]+"_meta.pkl"
            pickle.dump(meta_predictions, open(meta_outfile, "wb"))

        results = squad_evaluate(examples, predictions)

        return results
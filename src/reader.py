####################################
#  Class and utils for the Reader  #
####################################
import os
from transformers import pipeline

cwd = os.getcwd()

# TODO: hardcode model paths that we trained 
MODEL_PATHS = {
    'default_bert_base_uncased': 'bert-base-uncased',
    'bert_base_uncased_squad1':
        cwd+"/src/models/bert/bert-base-uncased-tuned-squad-1.0",
    'bert_base_cased_squad2':
        cwd+"/src/models/bert/bert-base-cased-tuned-squad-2.0/"
}

# What does the READER need?
# -- model and tokenizer
# -- ability to chunk text 
# -- more sophisticated prediction method 

# What classes should the READER interact with? 
# -- EXAMPLE class: can have an optional "answers" for examples that have answers for validation
# -- FEATURES class: stores the model-ingestible features of that CONTEXT
# -- RESULTS class: handles the output logits? Might not be necessary
# -- PIPELINE class: the thing that connects the reader to the retriever

class Reader:
    def __init__(
        self, 
        model_name="twmkn9/distilbert-base-uncased-squad2",
        tokenizer_name="twmkn9/distilbert-base-uncased-squad2",
        top_n_per_doc=3,
        use_gpu=False,
        handle_impossible_answer=True
        ):
        
        self.model = pipeline('question-answering', model=model_name, tokenizer=tokenizer_name, device=int(use_gpu)-1)
        self.kwargs = {'topk':top_n_per_doc, 'handle_impossible_answer':handle_impossible_answer}

    def predict(self, question, documents):
        # instead of having to deal with all the nitty gritty details of converting examples into features 
        # and creating sensible predictions, we can instead focus on predicting answers over a collection of docs
        # as this is what we'll be getting from the retriever

        all_predictions = []
        for doc in documents:            
            inputs = {"question": question, "context": doc['text']}
            predictions = self.model(inputs, **self.kwargs)

            # store these for each document for ranking at the end
            for pred in predictions:
                if pred['answer']:
                    answer = {
                        'probability': pred['score'],
                        'answer_text': pred['answer'],
                        'start_index': pred['start'],
                        'end_index': pred['end'],
                        # TODO: we need to keep better track of which document this answer comes from
                        'doc_id': doc['id']
                    }
                    all_predictions.append(answer)
        
        # Once we have all possible answers from all documents, sort them and take the top_n
        # This method does not take into account how highly ranked a document is (doc might have a score too)
        best_predictions = sorted(all_predictions, key=lambda x: x["probability"], reverse=True)[: self.kwargs["topk"]]

        results = {
            'question': question,
            'answers': best_predictions
        }
        return results


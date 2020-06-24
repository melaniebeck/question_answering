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
        use_gpu=True,
        handle_impossible_answer=True
        ):
        
        self.use_gpu = use_gpu
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
            
            # we want the best prediction from each document
            if len(predictions) == 1:
                best = predictions
            else:
                best = predictions[0]

            answer = {
                    'probability': best['score'],
                    'answer_text': best['answer'],
                    'start_index': best['start'],
                    'end_index': best['end'],
                    'doc_id': doc['id'],
                    'title': doc['title']
                }
            all_predictions.append(answer)

        # Simple heuristic: 
        # If the best prediction from each document is the null answer, return null
        # Otherwise, return the highest scored non-null answer
        null = True
        for prediction in all_predictions:
            if prediction['answer_text']:
                null = False
        
        if not null:
            # pull out only non-null answers
            all_predictions = [prediction for prediction in all_predictions if prediction['answer_text']]

        # sort predictions and return the highest ranked answer
        best_predictions = sorted(all_predictions, key=lambda x: x["probability"], reverse=True)[: self.kwargs["topk"]]
        results = {
            'question': question,
            'answers': best_predictions
        }
        return results


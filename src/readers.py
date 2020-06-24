####################################
#  Class and utils for the Reader  #
####################################
# collapse-hide
import os
from transformers import pipeline

class Reader:
    def __init__(
        self, 
        model_name="twmkn9/distilbert-base-uncased-squad2",
        tokenizer_name="twmkn9/distilbert-base-uncased-squad2",
        use_gpu=True,
        topk=3,
        handle_impossible_answer=True
        ):
        
        self.use_gpu = use_gpu
        self.model = pipeline('question-answering', model=model_name, tokenizer=tokenizer_name, device=int(use_gpu)-1)
        self.kwargs = {'handle_impossible_answer':handle_impossible_answer, 'topk':topk}

    def predict(self, question, documents, topk=None):
        """
        Compute text prediction for a question given a collection of documents

        Inputs:
            question: str, question string
            documents: list of document dicts, each with the following format:
                    {
                        'text': context string,
                        'id': document identification,
                        'title': name of document 
                    }
            topk (optional): int, if provided, overrides default topk
        
        Outputs:
            results: dict with the following format:
                {
                    'question': str, question string,
                    'answers': list of answer dicts, each including text answer, probability, 
                                start and end positions, and document metadata
                }
        """
        if topk:
            self.kwargs['topk'] = topk

        all_predictions = []
        for doc in documents:            
            inputs = {"question": question, "context": doc['text']}
            predictions = self.model(inputs, **self.kwargs)
            
            # we want the best prediction from each document
            if self.kwargs['topk'] == 1:
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
            # pull out and sort only non-null answers
            non_null_predictions = [prediction for prediction in all_predictions if prediction['answer_text']]
            sorted_non_null = sorted(non_null_predictions, key=lambda x: x['probability'], reverse=True)
            
            # append the null answers for completeness
            null_predictions = [prediction for prediction in all_predictions if not prediction['answer_text']]
            best_predictions = sorted_non_null + null_predictions
        else:  
            # sort null answers for funsies
            best_predictions = sorted(all_predictions, key=lambda x: x["probability"], reverse=True)[: self.kwargs["topk"]]
        
        results = {
            'question': question,
            'answers': best_predictions
        }
        return results

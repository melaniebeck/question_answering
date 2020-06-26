####################################
#  Class and utils for the Reader  #
####################################
import os
from transformers import pipeline

class Reader:
    def __init__(
        self, 
        model_name="twmkn9/distilbert-base-uncased-squad2",
        tokenizer_name="twmkn9/distilbert-base-uncased-squad2",
        use_gpu=True,
        handle_impossible_answer=True
        ):
        
        self.use_gpu = use_gpu
        self.model = pipeline('question-answering', model=model_name, tokenizer=tokenizer_name, device=int(use_gpu)-1)
        self.kwargs = {'handle_impossible_answer':handle_impossible_answer}

    def predict(self, question, documents, topk=None):
        # instead of having to deal with all the nitty gritty details of converting examples into features 
        # and creating sensible predictions, we can instead focus on predicting answers over a collection of docs
        # as this is what we'll be getting from the retriever
        self.kwargs['topk'] = topk

        all_predictions = []
        for doc in documents:            
            inputs = {"question": question, "context": doc['text']}
            predictions = self.model(inputs, **self.kwargs)
            print(predictions)

            # we want the best prediction from each document
            if topk == 1:
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

    def predict_combined(self, question, documents, topk=None):
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

        # combine all documents together into one long context
        context = ''
        for doc in documents:
            context += doc['text'] + ' '

        all_predictions = []
        inputs = {"question": question, "context": context}
        predictions = self.model(inputs, **self.kwargs)
        
        # we want the best prediction from each document
        if self.kwargs['topk'] == 1:
            predictions = [predictions]

        for pred in predictions:
            answer = {
                    'probability': pred['score'],
                    'answer_text': pred['answer'],
                    'start_index': pred['start'],
                    'end_index': pred['end'],
                    #'doc_id': doc['id'],
                    #'title': doc['title']
                }
            all_predictions.append(answer)

        # sort and truncate predictions
        best_predictions = sorted(all_predictions, key=lambda x: x["probability"], reverse=True)[: self.kwargs["topk"]]
        
        results = {
            'question': question,
            'answers': best_predictions
        }
        return results

    def predict_full_wiki(self, question, context, topk=None):
        """
        Andrew -- I modified one of the predict methods to more easily pass Squad-style examples
                The other methods expect input from a retriever so there's a lot of highly-
                specific hardcoded keywords, etc. Adjust this however you need!

        Inputs:
            question: str, question string
            context: str, content of an article
        """
        self.kwargs['topk'] = topk
        # since you'll be removing no-answer questions, this flag should be False
        self.kwargs['handle_impossible_answer'] = False

        inputs = {"question": question, "context": context}
        predictions = self.model(inputs, **self.kwargs)
        if topk == 1:
            return predictions
        
        # for evaluating on squad, all you really need is the top prediction
        return predictions[0]
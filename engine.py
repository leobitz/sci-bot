import json
from typing import List
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline
from summarizer.sbert import SBertSummarizer


class ChatBotEngine:

    def __init__(self) -> None:
        self.SUMMARIZER_MODEL = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        self.SENTENCE_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
        self.Q_ANSWERER = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
        self.REPHRASER = self.SUMMARIZER_MODEL
        self.CORPUS_DB, self.CORPUS_EMB = self.prepare_doc_encoding('data/val.txt', 10, 128)
        


    def bert_summarize_text(self, texts: List[str], max_word_length: int) -> str:
        result = self.SUMMARIZER_MODEL(texts, min_length=max_word_length)
        return result

    def prepare_doc_encoding(self, file_name: str, num_samples: int, window_size: int = 128) -> List:
        processed = []
        file = open(file_name, encoding='utf-8', mode='r')
        raw_articles = []
        for i in range(num_samples):
            art = json.loads(file.readline())
            passage = " ".join(art['article_text'])
            raw_articles.append(passage)

            if len(passage.split()) > window_size:
                passage = self.bert_summarize_text(passage, max_word_length=window_size)
            
            processed.append(passage)

        file.close()
        
        doc_vec = self.SENTENCE_ENCODER.encode(processed)

        return raw_articles, doc_vec


    def search(self, question: str, corpus_emb: np.ndarray,  top_k: int = 5) -> str:
        question_emb = self.SENTENCE_ENCODER.encode(question)
        hits = util.semantic_search(question_emb, corpus_emb, top_k=top_k)
        return hits

    def answer(self, question: str, context: str) -> str:
        result = self.Q_ANSWERER(question=question, context=context)
        return result


    def rephrase_explanation(self, final_context, exp_sentences=3):
        new_explanation = self.REPHRASER(final_context, num_sentences=exp_sentences)
        return new_explanation

    def query(self, question: str, explain = False, rephrase_explain = False, exp_sentences = 3, confidence_threshold = 0.5):

        hits = self.search(question, self.CORPUS_EMB)

        if hits[0][0]['score'] >= confidence_threshold:
            context = self.CORPUS_DB[hits[0][0]['corpus_id']]
            result = self.answer(question, context)
            
            exp = None

            score, ans = result['score'], result['answer']
            if explain:
                exp = context
                if rephrase_explain:
                    exp = self.rephrase_explanation(exp, exp_sentences= exp_sentences)
                ans = f"{ans} \n Explanation: {exp}"
            
            print(score)
            if score > confidence_threshold:
                return ans
        
        return "Sorry, I don't know" 

engine = ChatBotEngine()

question = "what is neurologists?"
ans = engine.query(question, True, True, exp_sentences=2, confidence_threshold=0.5)

print(ans)
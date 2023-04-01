import json
from typing import List
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline
from summarizer.sbert import SBertSummarizer

from top2vec import Top2Vec


class SlidingWindowDoc2Vec:

    def __init__(self, encoder, window_size=256, stride=128) -> None:
        self.window_size = 256
        self.stride = 128
        self.encoder = encoder
    
    def encode(self, text):
        words = text.split()

        if len(words) <= self.window_size:
            return self.encoder([text])
        
        windows = []
        for i in range(0, len(words) - self.stride, self.stride):
            sub_text = " ".join(words[i:i + self.window_size])
            windows.append(sub_text)

        return self.encoder(windows).mean(axis=0)


class ChatBotEngine:

    def __init__(self, 
                document_file_path,
                doc2vec = 'summarize', 
                 explain = False, 
                 rephrase_explain = False, 
                 exp_sentences = 3, 
                 confidence_threshold = 0.5, 
                 max_num_samples = 10,
                 window_size: int = 128, 
                 stride: int = 128) -> None:
        self.explain = explain
        self.rephrase_explain = rephrase_explain
        self.exp_sentences = exp_sentences
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.stride = stride
        self.max_num_samples = max_num_samples

        self.SUMMARIZER_MODEL = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        self.SENTENCE_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
        self.Q_ANSWERER = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
        self.REPHRASER = self.SUMMARIZER_MODEL
        self.topic_model = None

        if doc2vec == 'summarize':
            self.doc2vec_encode = self.SENTENCE_ENCODER.encode
        elif doc2vec == 'ave_window':
            encoder = SlidingWindowDoc2Vec(self.SENTENCE_ENCODER.encode, self.window_size, self.stride)
            self.doc2vec_encode = encoder.encode
        else:
            raise Exception("doc2vec strategy not specified")

        self.CORPUS_DB, self.CORPUS_EMB = self.prepare_doc_encoding(document_file_path, self.max_num_samples)

    def bert_summarize_text(self, texts: List[str], max_word_length: int) -> str:
        result = self.SUMMARIZER_MODEL(texts, min_length=max_word_length)
        return result

    def prepare_top_vec(self, file_name, num_samples: int):
        file = open(file_name, encoding='utf-8', mode='r')
        raw_articles = []
        
        for i in range(num_samples):
            art = json.loads(file.readline())
            passage = " ".join(art['article_text'])
            raw_articles.append(passage)

        file.close()
        self.topic_model = Top2Vec(raw_articles, embedding_model="all-MiniLM-L6-v2")

    def prepare_doc_encoding(self, file_name: str, num_samples: int) -> List:
        file = open(file_name, encoding='utf-8', mode='r')
        raw_articles = []
        doc_vec = []
        
        for i in range(num_samples):
            art = json.loads(file.readline())
            passage = " ".join(art['article_text'])
            raw_articles.append(passage)

            if len(passage.split()) > self.window_size:
                passage = self.bert_summarize_text(passage, max_word_length=self.window_size)
            
            enc = self.doc2vec_encode(passage)
            doc_vec.append(enc)

        file.close()
        
        doc_vec = np.stack(doc_vec)

        return raw_articles, doc_vec


    def search(self, question: str, corpus_emb: np.ndarray,  top_k: int = 5) -> str:
        if self.topic_model:
            hits = self.topic_model.query_documents(question, 1)
        else:
            question_emb = self.SENTENCE_ENCODER.encode(question)
            hits = util.semantic_search(question_emb, corpus_emb, top_k=top_k)
        return hits, question_emb

    def answer(self, question: str, context: str) -> str:
        result = self.Q_ANSWERER(question=question, context=context)
        return result

    def rephrase_explanation(self, final_context):
        new_explanation = self.REPHRASER(final_context, num_sentences=self.exp_sentences)
        return new_explanation

    def search_within_passage(self, question_emb, context, top_k=1):
        words = context.split()

        if len(words) <= self.window_size:
            return self.encoder([context])
        
        windows = []
        for i in range(0, len(words) - self.stride, self.stride):
            sub_text = " ".join(words[i:i + self.window_size])
            windows.append(sub_text)

        chunk_emb = self.SENTENCE_ENCODER.encode(windows)
        hits = util.semantic_search(question_emb, chunk_emb, top_k=top_k)
        
        return hits, windows


    def query(self, question: str):

        if question == None or question.strip() == '' or len(question.split()) <= 2:
            return "Please give me a question with at least 3 words"

        hits, question_emb = self.search(question, self.CORPUS_EMB)

        # if a document that is similar to the question intent is found
        if hits[0][0]['score'] >= self.confidence_threshold: 
            
            context = self.CORPUS_DB[hits[0][0]['corpus_id']]
            # within_passage_hit, chunks = self.search_within_passage(question_emb, context, top_k=5)
            # context = chunks[within_passage_hit[0][0]['corpus_id']]

            result = self.answer(question, context)
            
            exp = None

            score, ans = result['score'], result['answer']
            if self.explain:
                exp = context
                if self.rephrase_explain:
                    exp = self.rephrase_explanation(exp)
                ans = f"{ans} \n Explanation: {exp}"
            
            print(score)
            if score > self.confidence_threshold:
                return ans
        
        return "Sorry, I don't know" 

if __name__ == "__main__":
    engine = ChatBotEngine(document_file_path='data/val.txt', max_num_samples=100)

    question = "who decides whether stroke status is correct?"
    ans = engine.query(question)

    print(ans)

    # text = "who decides ChatBotEngine. whether stroke . ChatBotEngine status is correct?"
    # sentences = []
    # while True:

    #     text.index('.')
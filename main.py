import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
from engine import ChatBotEngine
import yaml


with open('botconfig.yaml', 'r') as file:
    botConfig = yaml.safe_load(file)
    chatbot_engine = ChatBotEngine(
        document_file_path=botConfig['corpus_file'],
        doc2vec = 'chunk_mean', 
        explain = botConfig['include_explanation'], 
        rephrase_explain = botConfig['rephrase_explanation'],
        exp_sentences = botConfig['num_explanation_sentence'], 
        confidence_threshold = botConfig['answer_score_threshold'], 
        max_num_samples = botConfig['max_corpus_samples'],
        window_size = botConfig['doc2vec_window_size'],
        stride = botConfig['doc2vec_stride_size'],
        topic_search= botConfig['topic_search']
    )
############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    pass
    


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = chatbot_engine.query(text)
        output.append(response)

    return SimpleText(dict(text=output))

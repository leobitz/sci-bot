## **Simple Scientific Q&A Chatbot**

## Architecture

The architecture includes the following major components:

- **Document Encoder** - Documents have different sizes and each document can be represented with a fixed-size vector. I used two options
  - BERT Summarizer: Summarize any text to a fixed token size text (256) and apply sentence encoder (BERT-Sentence Encoder)
  - Chunking: Apply the BERT sentence encoder on fixed-size chunks of the document and finally average them to get the final fixed-size representation

Chunking is faster since it's easier to process short-length documents than long sizes in Transformers

- **Document Retrieval -** Given a question, a search operation is applied to the corpus to find the right context. I implemented two methods
  - **Linear Search -** on the document vector space. Very slow
  - **Hierarchical Search** - Using Topic2Vec, we can build a tree and search on clusters rather than search on an array

- **Question Answerer -** Once the right context is found, a seq2seq model can be fed the question and the context. It will produce the answer.

- **Rephraser -** A small explanation can be provided along with the answer. As such, the context can be rephrased to a small text with a specified amount of sentences

## Dataset

[Pubmed Scientific Dataset](https://huggingface.co/datasets/scientific_papers) - I used this database. It has 119k samples on medical topics

## Improvements

Several improvements can be made to make the system more useable

- Extract scientific conversation from the [Stanford SHP dataset](https://huggingface.co/datasets/stanfordnlp/SHP) and fine-tune the whole system using RLHF. This will make answers more humanly
- Using the same backbone model for encoding documents, summarization, and answer inference. This will result in a massive memory footprint reduction
- A context might be long and answer generation inference can take longer time. Such an operation can be minimized using chunked search. I have applied that already, but the performance is not that great. With the right hyperparameter (window size) search, inference can be boosted.
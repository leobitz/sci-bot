FROM openfabric/openfabric-pyenv:0.1.9-3.8

RUN mkdir cognitive-assistant
WORKDIR /cognitive-assistant
COPY . .
RUN pip install -r requirements.txt

RUN sudo apt-get install unzip
RUN wget https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip
RUN unzip file.zip -d pubmed-dataset.zip

RUN poetry install -vvv --no-dev
EXPOSE 5000
CMD ["sh","start.sh"]
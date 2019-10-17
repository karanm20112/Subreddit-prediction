"""
Name: Karan pankaj Makhija and Jeet thakur
Version : Python 2.7
Title: LDA using Gensim
"""
import spacy
import pandas as pd
from spacy.lang.en import English
import nltk
import gensim
import pickle
#Load the english dictionary
spacy.load('en')
parser = English()
data = pd.read_json('Final1.json')
data_attr = data.iloc[:,1]
print(data_attr)
text_list = list(data_attr)
#Convert it to text
text = ''.join(text_list)

text = text[0:1000000]
def tokenize(text):
    """
    Tokenizing and cleaning the text
    :param text: the text which is to be converted
    :return: The list
    """
    lda = []
    tokens = parser(text)
    for token in tokens:
        #Removing Spaces
            if token.orth_.isspace():
                continue
                #Removing URL
            elif token.like_url:
                lda.append('URL')
                #Remove email id
            elif token.orth_.startswith('@'):
                lda.append('SCREEN_NAME')
            else:
                #Convert the text to lower case
                lda.append(token.lower_)
    return lda


nltk.download('wordnet')
from nltk.corpus import wordnet as wn
#get meaning of words, synonyms and antonymns
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

#we use wordnetlemmatizer to get the root
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
#Filtering stop words
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    """
    Prepare the text
    :param text: The text to be prepared
    :return: the text
    """
    #Tokenize, remove the stop worda
    lda = tokenize(text)
    lda = [token for token in lda if len(token) > 4]
    lda = [token for token in lda if token not in en_stop]
    lda = [get_lemma(token) for token in lda]
    return lda

import random
text_data = []
print(data_attr)
for i,row in enumerate(data_attr.iteritems()):
    tokens = prepare_text_for_lda(str(row))
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)


#make a dictionary
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
#Dump in corpus
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
#10 topics to be extracted
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

#Plot the corpus dictionary using PYLDAviz
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
print(dictionary)
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
#Display the LDA
pyLDAvis.show(lda_display)




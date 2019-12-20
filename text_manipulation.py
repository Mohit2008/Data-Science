#1.)Working with text using BAG OF WORDS model
#link-https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
vect = CountVectorizer()
vect.fit(list_of_messages)
dtm = vect.transform(list_of_messages)
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())

#2.)Working with text using Term Frequency Inverse documnet frequency model
# args-max_features , min_df ,max_df ,stop_words 
#link-https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/
from sklearn.feature_extraction.text import TfidfVectorizer
def createDTM(list_of_messages):
    vect = TfidfVectorizer()
    dtm = vect.fit_transform(list_of_messages) # create DTM
    # create pandas dataframe of DTM
    return pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 

#3.)Working with text using Word2Vec model
#link-https://taylorwhitten.github.io/blog/word2vec
import gensim
from gensim.models import Word2Vec
from gensim.models import Phrases
model = word2vec.Word2Vec(list_of_sentences_containing_list_of_tokens, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = "nytimes_oped"
model.save(model_name)
new_model = gensim.models.Word2Vec.load('nytimes_oped')

#Adding support for bigram and tiagram
bigramer = gensim.models.Phrases(sentences)
trigram = Phrases(bigram[sentence_stream])
model = Word2Vec(trigram, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)



#4.)Working with text using Word2Vec model and using that for text classification by buidling a doc vector
#http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)
model = gensim.models.Word2Vec(X, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
         

etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

# 5 perform tokenisation using spacy module

import spacy
nlp = spacy.load('en_core_web_sm') # Load the en_core_web_sm model
doc = nlp(gettysburg)
tokens = [token.text for token in doc]# Generate the tokens

or

lemmas = [token.lemma_ for token in doc] # Generate lemmas for the text that has been passed

or

# Remove stopwords and non-alphabetic tokens, and use lemmitisation
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

pos = [token.pos_ for token in doc] # get the pos tags


# 6 Function to preprocess text

import spacy
nlp = spacy.load('en_core_web_sm') # Load the en_core_web_sm model
# Function to preprocess text
def preprocess(text):
    doc = nlp(text, disable=['ner', 'parser']) # Create Doc object
    lemmas = [token.lemma_ for token in doc] # Generate lemmas
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords] # Remove stopwords and non-alphabetic characters
    return ' '.join(a_lemmas)
  

ted['transcript'] = ted['transcript'].apply(preprocess) # Apply preprocess to ted['transcript']

# 7 In NLP you can create new features by counting no of chars, counting no of words, get the average word length, count no of
nouns, count no of pronoun 

# do named entite recognition
import spacy
nlp = spacy.load('en') # install 'en' model (python3 -m spacy download en)
doc = nlp("Alphabet is a new startup in China")
print('Name Entity: {0}'.format(doc.ents))
for ent in doc.ents:
  print(ent.text, ent.labels_)
 
 
 
# document similarity
doc1 = nlp(u'Hello this is document similarity calculation')
doc2 = nlp(u'Hello this is python similarity calculation')
print (doc1.similarity(doc2)) 
#Tokenisation
for token in doc1:
    print(token.text)
    print (token.vector)
# Word vector
print (token.vector)   #-  prints word vector form of token. 
print (doc1[0].vector) #- prints word vector form of first token of document.
print (doc1.vector)    #- prints mean vector form for doc1


#Buidling chatbots with RASA , here we will convert free from text to structured data with intents and entities

from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
args = {"pipeline": "spacy_sklearn"}# Create args dictionary
config = RasaNLUConfig(cmdline_args=args)# Create a configuration and trainer
trainer = Trainer(config)
training_data = load_data("./training_data.json")# Load the training data
interpreter = trainer.train(training_data)# Create an interpreter by training the model
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))# Try it out

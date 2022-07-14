#  Imports

from termcolor import colored
from tqdm.notebook import tqdm
from utils import unpickler, pickler

# !python -m spacy download en_core_web_sm
# !python -m spacy download en_core_web_md
# !python -m spacy download xx_sent_ud_sm

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


# Load spacy models
# Add LanguageDetector and assign it a string name
@Language.factory("language_detector")
def create_language_detector(nlp, name):
	return LanguageDetector(language_detection_function=None)


mult_nlp = spacy.load('xx_sent_ud_sm')
mult_nlp.add_pipe('language_detector', last=True)
nlp = spacy.load('en_core_web_md')
nlp.disable_pipe('parser')
nlp.disable_pipe('ner')

valid_POS = {'VERB', 'NOUN', 'ADJ', 'PROPN'}
specific_stw = {'relevant', 'simple', 'base'}


# ------------------------
# SPACY
# ------------------------


def text_preprocessing(rawtext, stop_words: list):
	mult_doc = mult_nlp(rawtext)
	english_text = ' '.join([sent.text for sent in mult_doc.sents if sent._.language['language'] == 'en'])
	doc = nlp(english_text)
	lemmatized = ' '.join([token.lemma_ for token in doc
						   if token.is_alpha
						   and token.pos_ in valid_POS
						   and not token.is_stop
						   and not token.is_punct
						   and not token.like_num
						   and token.lemma_ not in stop_words
						   and token.lemma_ not in specific_stw])
	return lemmatized


# ------------------------
# SPACY PIPELINE
# ------------------------

def spacy_pipeline(corpus_file: str, corpus_name: str):
	"""Preprocess with SpaCy given a pickled corpus dictionary"""
	# Unpickle
	data = unpickler(corpus_file)
	print(f'Source: {corpus_file}, length: {len(data)} texts')

	# Load stopwords
	stop_words = unpickler('stopwords.pkl').get(corpus_name)

	# Turn dict into list of strings:
	mycorpus = {key: text_preprocessing(str(el), stop_words).strip().split() for key, el in tqdm(data.items(),
																								 desc="Preprocessing texts")}
	print(colored('Number of documents in corpus: ' + str(len(mycorpus)), 'green'))

	# Save preprocessed corpus list
	pickler(f'corpus/{corpus_name}_corpus.pkl', list(mycorpus.values()))

	# Save preprocessed corpus to .txt file for MALLET
	with open(f'corpus/{corpus_name}_corpus.txt', 'w') as f:
		for key, doc in mycorpus.items():
			print(f"{key} 0 {' '.join(doc)}", file=f)

	print(f'Saved to: corpus/{corpus_name}_corpus.pkl & corpus/{corpus_name}_corpus.txt')
	return

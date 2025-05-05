import os
import spacy
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from thinc.api import set_gpu_allocator, require_gpu

# Keep important words for sentiment analysis
KEEP_WORDS = { 
	"not", "no", "nor", "never", "none", "nobody", "nowhere", "nothing", "neither",
	"hardly", "barely", "scarcely", "damn", "very", "really", "quite", "so", "too", 
	"enough", "just", "even", "still", "only", "but", "although", "though", "however"
}

nlp = None
custom_stop_words = STOP_WORDS - KEEP_WORDS


def read_texts(data_folder, given_folder=None):
	contents = []
	
	folder_path = os.path.join("data", data_folder)

	if given_folder:
			folder_path = os.path.join(folder_path, given_folder)

	if not os.path.exists(folder_path):
			print(f"Folder not found: {folder_path}")
			return contents

	for file in os.listdir(folder_path):
			if file.endswith(".txt"):
					file_path = os.path.join(folder_path, file)
					with open(file_path, 'r', encoding='utf-8') as f:
							content = f.read()
							contents.append(content)
	return contents


def set_nlp(model_name):
	global nlp
	if model_name.endswith("trf"):
			set_gpu_allocator("pytorch")
			require_gpu(0)
	nlp = spacy.load(model_name)
	return nlp

def get_nlp(language_model):
	global nlp
	if nlp is None:
			nlp = set_nlp(language_model)
	return nlp

def check_token(token):
	return bool(
			token
			and str(token).strip()
			and token.is_alpha
			and not token.is_punct
			and token.text.lower() not in custom_stop_words
	)

def get_lemma(token):
	return token.lemma_.strip()

def extract_embeddings(docs):
	embeddings = []
	for doc in docs:
			# Check if the text is empty, fill empty with a zero vector
		if len(doc) == 0:
			embeddings.append(np.zeros(768))
		else:
				# Use tokens that have vectors, and take the mean of those embeddings
				valid_tokens = [token.vector for token in doc if token.has_vector]
				if valid_tokens:
					doc_embedding = np.mean(valid_tokens, axis=0)
				else:
					# If no valid tokens, append a zero vector (this should rarely happen)
					doc_embedding = np.zeros(768)
				embeddings.append(doc_embedding)

	print(embeddings[0].shape if embeddings else 'empty')
	return embeddings

def preprocess_text_embeddings(data_folder, given_folder=None, language_model="en_core_web_trf"):
	texts = read_texts(data_folder, given_folder)
	nlp_instance = get_nlp(language_model)
	docs = [nlp_instance(text) for text in texts]
	embeddings = extract_embeddings(docs)
	return embeddings

def preprocess_text_bow(data_folder, given_folder=None, language_model="en_core_web_sm", lemmatize=True):
	texts = read_texts(data_folder, given_folder)
	nlp_instance = get_nlp(language_model)
	processed_texts = []

	for text in texts:
			text_lowered = text.lower()
			doc = nlp_instance(text_lowered)
			if lemmatize:
					processed_text = [get_lemma(token) for token in doc if check_token(token)]
			else:
					processed_text = [token.text for token in doc if check_token(token)]

			processed_texts.append(processed_text)
	return processed_texts

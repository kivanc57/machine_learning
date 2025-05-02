import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from thinc.api import set_gpu_allocator, require_gpu

# Keep important words for sentiment analysis
KEEP_WORDS = { 
    "not", "no", "nor", "never", "none", "nobody", "nowhere", "nothing", "neither",
    "hardly", "barely", "scarcely", "damn", "very", "really", "quite", "so", "too", 
    "enough", "just", "even", "still", "only", "but", "although", "though", "however"
}

LANGUAGE_MODEL = "en_core_web_sm"
nlp = None

custom_stop_words = STOP_WORDS - KEEP_WORDS


def read_texts(data_folder, given_folder):
  contents = []
  folder_path = os.path.join("data", data_folder, given_folder)

  for file in os.listdir(folder_path):
    if file.endswith(".txt"):
      file_path = os.path.join(folder_path, file)
      with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        contents.append(content)
  return contents

def set_nlp():
  global nlp
  if LANGUAGE_MODEL.endswith("trf"):
    set_gpu_allocator("pytorch")
    require_gpu(0)
  nlp = spacy.load(LANGUAGE_MODEL)
  return nlp

def get_nlp():
  global nlp
  if nlp is None:
    nlp = set_nlp()
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

def preprocess_text_bow(data_folder, given_folder, lemmatize=True):
    texts = read_texts(data_folder, given_folder)
    nlp_instance = get_nlp()
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

# preprocess text before passing through BERT

import re

# clean text from noise
def clean_text(text):
    # filter to allow only alphabets
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    
    # remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # convert to lowercase to maintain consistency
    text = text.lower()
       
    return text

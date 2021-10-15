import time
import numpy as np
import pandas as pd

from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from config import MAX_LEN


def preprocessing(df):
    sentences = df.sentence.values
    labels = np.array([l for l in df.label.values]) #np.array([labels_encoding[l] for l in df.label.values])

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    
    encoded_sentences = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
                            sent,
                            add_special_tokens = True,
                            truncation=True,
                            max_length = MAX_LEN
                    )
        
        encoded_sentences.append(encoded_sent)
    encoded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")
    return encoded_sentences, labels
    
def attention_masks(encoded_sentences):
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

def compute_accuracy(preds, labels):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()
    return np.sum(p==l)/len(l)
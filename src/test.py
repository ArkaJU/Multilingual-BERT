import os 
import time
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import random
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from utils import preprocessing, attention_masks, compute_accuracy
from config import BATCH_SIZE

def run_test(df_test, model_path):

    if torch.cuda.is_available():    
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")

    model = torch.load(model_path)
    test_encoded_sentences, test_labels = preprocessing(df_test)
    test_attention_masks = attention_masks(test_encoded_sentences)

    test_inputs = torch.tensor(test_encoded_sentences)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model.eval()
    eval_loss, eval_acc = 0,0
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        eval_data, eval_masks, eval_labels = batch
        with torch.no_grad():
            out = model(eval_data,
                        token_type_ids = None,
                        attention_mask=eval_masks)
        logits = out[0]
        logits = logits.detach().cpu().numpy()
        eval_labels = eval_labels.to('cpu').numpy()
        batch_acc = compute_accuracy(logits, eval_labels)
        eval_acc += batch_acc
    print(f"Accuracy: {eval_acc/(step+1)}")

if __name__ == "__main__":
    
    model_path = "/content/Multilingual-BERT/model/mbert.pt"

    print("Evaluating on english \n")
    df_test = pd.read_csv("datasets/MLDoc/english.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

    print("Evaluating on spanish \n")
    df_test = pd.read_csv("./spanish.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

    print("Evaluating on french \n")
    df_test = pd.read_csv("./french.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

    print("Evaluating on italian \n")
    df_test = pd.read_csv("./italian.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

    print("Evaluating on japanese \n")
    df_test = pd.read_csv("./japanese.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

    print("Evaluating on russian \n")
    df_test = pd.read_csv("./russian.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

    print("Evaluating on german \n")
    df_test = pd.read_csv("./german.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_test(df_test, model_path)

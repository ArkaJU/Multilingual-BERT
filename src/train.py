import os 
import time
import numpy as np
import pandas as pd

import torch
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig, BertPreTrainedModel, BertModel
from transformers import get_linear_schedule_with_warmup

from utils import preprocessing, attention_masks, compute_accuracy
from config import BATCH_SIZE, EPOCHS, MAX_LEN

class MBERT:
  def __init__(self):
    if torch.cuda.is_available():    
        self.device = torch.device("cuda")
        print('GPU in use:', torch.cuda.get_device_name(0))
    else:
        print('using the CPU')
        self.device = torch.device("cpu")

  def setup(self):

    # load the datasets
    df = pd.read_csv("datasets/MLDoc/english.train.10000", delimiter='\t', header=None, names=['label', 'sentence'])
    df_test = pd.read_csv("datasets/MLDoc/english.dev", delimiter='\t', header=None, names=['label', 'sentence'])
    
    train_encoded_sentences, train_labels = preprocessing(df)
    train_attention_masks = attention_masks(train_encoded_sentences)

    test_encoded_sentences, test_labels = preprocessing(df_test)
    test_attention_masks = attention_masks(test_encoded_sentences)

    train_inputs = torch.tensor(train_encoded_sentences)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_attention_masks)

    validation_inputs = torch.tensor(test_encoded_sentences)
    validation_labels = torch.tensor(test_labels)
    validation_masks = torch.tensor(test_attention_masks)

    # data loader for training
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # data loader for validation
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)
    

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    self.model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels = 4,   
        output_attentions = False, 
        output_hidden_states = False, 
    )

    self.model.cuda()

    self.optimizer = AdamW(self.model.parameters(),
                      lr = 3e-5, 
                      eps = 1e-8, 
                      weight_decay = 0.01
                    )

    total_steps = len(self.train_dataloader) * EPOCHS
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                num_warmup_steps = 0, # 10% * datasetSize/batchSize
                                                num_training_steps = total_steps)

  def run_train(self):
    losses = []
    for e in range(EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(e + 1, EPOCHS))
        start_train_time = time.time()
        total_loss = 0
        self.model.train()
        print("======== TRAIN ========")
        for step, batch in enumerate(tqdm(self.train_dataloader)):

            # if step%10 == 0:
            #     elapsed = time.time()-start_train_time
            #     print(f'{step}/{len(self.train_dataloader)} --> Time elapsed {elapsed}')

            # input_data, input_masks, input_labels = batch
            input_data = batch[0].to(self.device)
            input_masks = batch[1].to(self.device)
            input_labels = batch[2].to(self.device)

            self.model.zero_grad()

            # forward propagation
            out = self.model(input_data,
                        token_type_ids = None, 
                        attention_mask = input_masks,
                        labels = input_labels)
            
            loss = out[0]
            total_loss = total_loss + loss.item()

            # backward propagation
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)

            self.optimizer.step()
        
        epoch_loss = total_loss/len(self.train_dataloader)
        losses.append(epoch_loss)
        print(f"Training took {time.time()-start_train_time} seconds")

        # Validation
        start_validation_time = time.time()
        self.model.eval()
        eval_loss, eval_acc = 0,0
        print("======== VALIDATION ========")
        for step, batch in enumerate(tqdm(self.validation_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            eval_data, eval_masks, eval_labels = batch
            with torch.no_grad():
                out = self.model(eval_data,
                            token_type_ids = None, 
                            attention_mask=eval_masks)
            logits = out[0]

            #  Uncomment for GPU execution
            logits = logits.detach().cpu().numpy()
            eval_labels = eval_labels.to('cpu').numpy()
            batch_acc = compute_accuracy(logits, eval_labels)

            # Uncomment for CPU execution
            # batch_acc = compute_accuracy(logits.numpy(), eval_labels.numpy())

            eval_acc += batch_acc
        print(f"Accuracy: {eval_acc/(step+1)}, Time elapsed: {time.time()-start_validation_time}")
    return losses   

  def save(self):
      output_dir = './model_save'

      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
      
      model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
      model_to_save.save_pretrained(output_dir)

if __name__ == "__main__":
    mbert = MBERT()
    mbert.setup()
    losses = mbert.run_train()
    mbert.save()

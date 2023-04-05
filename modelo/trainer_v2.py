from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from untils import *
import torch
import json



with open('data/train.json', 'r') as f:
    train_data = json.load(f)


with open('data/test.json', 'r') as f:
    test_data = json.load(f)


tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
model = AlbertForQuestionAnswering.from_pretrained('albert-large-v2')

train_dataloader=extract_data(train_data,tokenizer)
test_dataloader=extract_data(test_data,tokenizer)

train(model,train_dataloader=train_dataloader,val_dataloader=test_dataloader)
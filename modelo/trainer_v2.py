from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from untils import *
import torch
import json
import os
ruta_a_modelos = os.path.join(os.path.dirname(__file__), 'modelos_entrenados')


with open('data/train.json', 'r') as f:
    train_data = json.load(f)


with open('data/test.json', 'r') as f:
    test_data = json.load(f)


tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
model = AlbertForQuestionAnswering.from_pretrained('albert-large-v2')



inputs,targets= extract_data(train_data)
inputs_test,targets_test= extract_data(test_data)

##########################################################################

input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
attention_masks = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["attention_mask"]
target_ids = tokenizer.batch_encode_plus(targets, padding=True, return_tensors="pt")["input_ids"]

##########################################################################

input_test_ids = tokenizer.batch_encode_plus(inputs_test, padding=True, return_tensors="pt")["input_ids"]
attention_test_masks = tokenizer.batch_encode_plus(inputs_test, padding=True, return_tensors="pt")["attention_mask"]
target_test_ids =tokenizer.batch_encode_plus(targets_test, padding=True, return_tensors="pt")["input_ids"]
##########################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(3):
    contador=0
    for batch_idx in range(0, len(input_ids),10 ):
        input_batch = input_ids[batch_idx:batch_idx+8].to(device)
        attention_mask_batch = attention_masks[batch_idx:batch_idx+8].to(device)
        target_batch = target_ids[batch_idx:batch_idx+8].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_batch, attention_mask=attention_mask_batch, start_positions=target_batch, end_positions=target_batch)

        loss = outputs[0]

        loss.backward()

        optimizer.step()

        perdida=loss.item()
        print('Epoch:', epoch, 'Batch:', batch_idx, 'Loss:', perdida,'Contador:',contador)

        if (contador == 0 or (contador%10 ==0)):
            ruta_al_modelo = os.path.join(ruta_a_modelos, f'trained_model_{epoch}_{batch_idx}_Lose_{perdida}')
            model.save_pretrained(ruta_al_modelo)
        
        contador=+1

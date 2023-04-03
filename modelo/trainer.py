from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
import os
ruta_a_modelos = os.path.join(os.path.dirname(__file__), 'modelos_entrenados')


with open('data/train.json', 'r') as f:
    train_data = json.load(f)


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


inputs = []
targets = []

for obj in train_data['data']:
    for paragraph in obj['paragraphs']:
        for qa in paragraph['qas']:
            input_text = paragraph['context'] + " " + qa['question']
            if not qa['is_impossible']:
                target_text = qa['answers'][0]['text']
            else:
                target_text = ''
                
            inputs.append(input_text)
            targets.append(target_text)

input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
target_ids = tokenizer.batch_encode_plus(targets, padding=True, return_tensors="pt")["input_ids"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(3):
    contador=0
    for batch_idx in range(0, len(input_ids), 8):
        input_batch = input_ids[batch_idx:batch_idx+8].to(device)
        target_batch = target_ids[batch_idx:batch_idx+8].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_batch, labels=target_batch)

        loss = outputs[0]

        loss.backward()

        optimizer.step()

        perdida=loss.item()
        print('Epoch:', epoch, 'Batch:', batch_idx, 'Loss:', perdida,'Contador:',contador)

        if (contador == 0 and (contador%10 ==0)):
            ruta_al_modelo = os.path.join(ruta_a_modelos, f'trained_model_{epoch}_{batch_idx}_Lose_{perdida}')
            model.save_pretrained(ruta_al_modelo)
        
        contador=+1


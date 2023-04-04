import torch
import os
ruta_a_modelos = os.path.join(os.path.dirname(__file__), 'modelos_entrenados')

def extract_data(data):
    inputs=[]
    targets=[]
    for obj in data['data']:
        for paragraph in obj['paragraphs']:
            for qa in paragraph['qas']:
                input_text = paragraph['context'] + " " + qa['question']
                if not qa['is_impossible']:
                    target_text = qa['answers'][0]['text']
                else:
                    target_text = ''
                    
                inputs.append(input_text)
                targets.append(target_text)

    return inputs,targets


def tokenizador(inputs,targets,tokenizer):
    input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
    attention_masks = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["attention_mask"]
    target_ids = tokenizer.batch_encode_plus(targets, padding=True, return_tensors="pt")["input_ids"]

    return input_ids, attention_masks,target_ids


def training(model,input_ids ,attention_masks,target_ids,input_test_ids ,attention_test_masks ,target_test_ids  ):
        
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

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import os
ruta_a_modelos = os.path.join(os.path.dirname(__file__), 'modelos_entrenados')


def tokenizador(inputs,targets,tokenizer):
    input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
    attention_masks = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["attention_mask"]
    target_ids = tokenizer.batch_encode_plus(targets, padding=True, return_tensors="pt")["input_ids"]

    return input_ids, attention_masks,target_ids



def extract_data(data,tokenizer):
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
                
    input_tensor,targets_tensor=tokenizador(inputs,targets,tokenizer)
    dataset=TensorDataset(input_tensor,targets_tensor)
    return DataLoader(dataset,batch_size=2)


def train(model, train_dataloader, val_dataloader, epochs=3, learning_rate=3e-5, save_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf') # inicializa mejor pérdida de validación como infinito

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if save_dir:
            if avg_val_loss < best_val_loss: # guarda el modelo sólo si su rendimiento en la validación es mejor que el mejor modelo anterior
                best_val_loss = avg_val_loss
                save_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"Saved model at {save_path}")

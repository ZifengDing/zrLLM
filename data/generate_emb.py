from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("t5-11b")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-11b")

print("model loaded")

def get_combinedEmb(dataset,input_file,model, tokenizer,output_path,batch_size=15):
    
    input_file_path='{}/{}'.format(dataset,input_file)

    list_of_tensors = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for id, line in enumerate(f): 
            input_text = line.strip()
            enc_inputs= tokenizer(input_text, max_length=150, return_tensors="pt", padding="max_length", truncation=True)

            # Pass the tokenized input through the T5 model to get the last_hidden_state
            # forward pass through encoder only
            outputs = model.encoder( 
                input_ids=enc_inputs["input_ids"], 
                attention_mask=enc_inputs["attention_mask"], 
                return_dict=True
            )
            # get the final hidden states
            emb = outputs.last_hidden_state
            list_of_tensors.append(emb)
            if len(list_of_tensors) >= batch_size:
                print(f"Processing batch {(id+1) // batch_size}")
                
                concatenated_tensor = torch.cat(list_of_tensors, dim=0)
                torch.save(concatenated_tensor, f'{output_path}_batch{(id+1) // batch_size}.pt')
                concatenated_numpy_array = concatenated_tensor.detach().numpy()
                np.save(f'{output_path}_batch{(id+1) // batch_size}.npy', concatenated_numpy_array)
                # Clear the list for the next batch
                list_of_tensors = []
        # After processing all items, save any remaining embeddings
        if list_of_tensors:
            #tensor_number += len(list_of_tensors)
            print(f"Processing the remaining batch")
            concatenated_tensor = torch.cat(list_of_tensors, dim=0)
            torch.save(concatenated_tensor, f'{output_path}_batch{(id+1) // batch_size + 1}.pt')
            concatenated_numpy_array = concatenated_tensor.detach().numpy()
            np.save(f'{output_path}_batch{(id+1) // batch_size + 1}.npy', concatenated_numpy_array)
               

        print("All embeddings of {} saved".format(dataset))

get_combinedEmb("ACLED",input_file='relation_explanation.txt',model=model, tokenizer=tokenizer,output_path='ACLED/t5_11b/Relation_Expl_Embedding',batch_size=23)
get_combinedEmb("ICEWS21",input_file='relation_explanation.txt',model=model, tokenizer=tokenizer,output_path='ICEWS21/t5_11b/Relation_Expl_Embedding',batch_size=15)
get_combinedEmb("ICEWS22",input_file='relation_explanation.txt',model=model, tokenizer=tokenizer,output_path='ICEWS22/t5_11b/Relation_Expl_Embedding',batch_size=15)


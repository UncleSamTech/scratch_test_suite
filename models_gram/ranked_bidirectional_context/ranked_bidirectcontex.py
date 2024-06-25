import os
from transformers import BertTokenizer
from datasets import load_dataset
import torch
import pickle
from transformers import BertTokenizer,BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments,BertForMaskedLM


class bidirectional_context:

    def __init__(self):
        self.forward_token_context = None
        self.backword_token_context = None
        self.missing_token = None
        
    

    
    def load_dataset_meth(self,dataset_path):
        return load_dataset(dataset_path,data_files={"train":"scratch_train_data_90.txt"})
    
    def tokenize_dataset(self,data_path,dataset_path,result_path):
        #load the Scratch3 dataset
        lines  = None
        dataset = self.load_dataset_meth(data_path)

        #load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        with open(dataset_path,'r',encoding='utf-8') as f:
            linesval = f.readlines()
            def tokenize_function(val):
                val = linesval
                max_len_ov = max([len(each_line) for each_line in val])
                batch_sample = int(max_len_ov/2)
                sample_data = {'text':[line.strip() for line in val]}
                
                
            
                val = tokenizer(sample_data['text'],padding="max_length",truncation=True,max_length=128)
                
                return val   
        
        tokenized_datasets = dataset.map(tokenize_function,batched=True)
        
        with open(f"{result_path}tokenized_ranked_bidirectional_context_tokenizer.pickle","wb") as tk:
                pickle.dump(tokenizer,tk,protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f"{result_path}tokenized_ranked_bidirectional_context_tokenized_datasets.pickle","wb") as tk:
                pickle.dump(tokenized_datasets,tk)

        
        return tokenized_datasets,tokenizer
    
    def training_data(self,data_path,dataset_path,result_path):
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        linesval = None
        tokenizer_data = None
        tokenizer_tok = None
        #tokenizer_d,tokenizer_val=self.tokenize_dataset(data_path,dataset_path,result_path)
        
        with open(f"{result_path}tokenized_ranked_bidirectional_context_tokenized_datasets.pickle","rb") as tk:
                tokenizer_data = pickle.load(tk)

        with open(f"{result_path}tokenized_ranked_bidirectional_context_tokenizer.pickle","rb") as tk2:
                tokenizer_tok = pickle.load(tk2)

        with open(dataset_path,'r',encoding='utf-8') as f:
            linesval = f.readlines()

        
        def custom_data_collator(dictionary_data):
            dictionary_data = linesval

            max_length = max([len(each_line) for each_line in dictionary_data])
            
            input_ids = []
            attention_masks = []
            labels = []
            total_length = len(dictionary_data)
            batch_length = int(total_length / 2)
            print(batch_length)

            # Example usage with a tokenizer
            #tokenizer_see = BertTokenizer.from_pretrained('bert-base-uncased')
            sample_data = [{"input_ids":tokenizer_tok.encode(each_data,add_special_token=True),'attention_mask': [1]*len(tokenizer_tok.encode(each_data, add_special_tokens=True))} for each_data in dictionary_data[0:batch_length]]
            
            # Create a DataLoader with a batch size of 2
            #data_loader = DataLoader(sample_data, batch_size=batch_length, collate_fn=lambda x: custom_data_collator(x, tokenizer))
            #for batch in data_loader:
                 #print(batch)

            for vd in sample_data:
                input_id =  vd['input_ids']

                attention_mask = vd['attention_mask']

                label = input_id.copy()

                if len(input_id) > 1:
                    label[1] = tokenizer_tok.mask_token_id
                    input_id[1] = tokenizer_tok.mask_token_id
                    input_ids.append(torch.tensor(input_id))
                    attention_masks.append(torch.tensor(attention_mask))
                    labels.append(torch.tensor(label))
                
                elif len(input_id) == 1:
                     label[0] = tokenizer_tok.mask_token_id
                     input_id[0] = tokenizer_tok.mask_token_id
                     input_ids.append(torch.tensor(input_id))
                     attention_masks.append(torch.tensor(attention_mask))
                     labels.append(torch.tensor(label))
                

            input_ids = pad_sequence(input_ids,batch_first=True,padding_value=tokenizer_tok.pad_token_id)
            attention_masks = pad_sequence(attention_masks,batch_first=True,padding_value=0)
            labels = pad_sequence(labels,batch_first=True,padding_value=-100)

            return {"input_ids":input_ids,"attention_mask":attention_masks,"labels":labels}
        
        

        training_args = TrainingArguments(
            output_dir='/Users/samueliwuchukwu/documents/output_dir',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_100,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=500
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=custom_data_collator,
            train_dataset=tokenizer_data['train'])
        
        trainer.train()

        model.save_pretrained("/Users/samueliwuchukwu/documents/saved_model")
        tokenizer_tok.save_pretrained("/Users/samueliwuchukwu/documents/saved_model")
        
    def predict_top5(self,model_path,test_data):
         
         #load the model
         model= BertForMaskedLM.from_pretrained(model_path)
         tokenizer = BertTokenizer.from_pretrained(model_path)
         model.eval()

         with open(test_data,"r",encoding="utf-8",errors="ignore") as td:
              lines = td.readlines()

               #mask all tokens at index 1 provided the sentence length is more that 1
               #define a custom data collator for this

         def custom_data_collator(self,dictionary_data):
            dictionary_data = lines
            input_ids = []
            attention_masks = []
            total_length = len(dictionary_data)
            batch_length = int(total_length / 2)
            sample_data = [{"input_ids":tokenizer.encode(each_data,add_special_token=True),'attention_mask': [1]*len(tokenizer.encode(each_data, add_special_tokens=True))} for each_data in dictionary_data[0:batch_length]]
            
            for each_data in sample_data:
                input_id = each_data['input_ids']
                attention_mask = each_data['attention_mask']

                if len(input_id) > 1:  
                    input_id[1] = tokenizer.mask_token_id

                    input_ids.append(torch.tensor(input_id))
                    attention_masks.append(torch.tensor(attention_mask))
                
                elif len(input_id) == 1:
                     input_id[0] = tokenizer.mask_token_id
                     input_ids.append(torch.tensor(input_id))
                     attention_masks.append(torch.tensor(attention_mask))
            
        
            input_ids = pad_sequence(input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)
            attention_masks = pad_sequence(attention_masks,batch_first=True,padding_value=0)
        
            return {"input_ids":input_ids,"attention_mask":attention_masks}

         test_data_sent = [{'input_ids': tokenizer.encode(sentence, add_special_tokens=True),
                  'attention_mask': [1]*len(tokenizer.encode(sentence, add_special_tokens=True))} 
                 for sentence in lines]
         
         test_batch = custom_data_collator(test_data_sent,tokenizer)

         #predict top five tokens

         with torch.no_grad():
              outputs = model(input_ids=test_batch["input_ids"],attention_mask=test_batch["attention_mask"])
              predictions = outputs.logits
        
            # Get the top 5 predictions for the masked token at index 1 for each sentence
         top_k = 5
         masked_index = 1
         predicted_tokens = []

         for i in range(predictions.shape[0]):
            softmax_scores = torch.softmax(predictions[i, masked_index], dim=-1)
            top_k_indices = torch.topk(softmax_scores, top_k).indices
            #top_k_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_k_indices]
            predicted_tokens.append(top_k_indices)
        
         #print(predicted_tokens)
         return predicted_tokens
    
    def reciprocal_rank(self,ground_truth, prediction):
        for rank, token in enumerate(prediction, start=1):
            if token.item() in ground_truth:
                return 1 / rank
        return 0
    
    def mrr_rank(self,model_path,test_data):
         ground_truth = []
         reciprocal_ranks = [self.reciprocal_rank(ground_truth,prediction) for prediction in self.predict_top5(model_path,test_data)]
         mrr = sum(reciprocal_ranks)/len(reciprocal_ranks)

         return mrr

bid = bidirectional_context()
#bid.tokenize_dataset("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/Users/samueliwuchukwu/documents/result_path")
#bid.training_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/Users/samueliwuchukwu/documents/result_path")
bid.predict_top5("/Users/samueliwuchukwu/documents/saved_model","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/scratch_test_data_10.txt")
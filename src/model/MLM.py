import time
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast,BertForMaskedLM,BertConfig
import numpy as np
import os
'''
1. Design model( input, output size, forward pass)
2. Construct loss and optimizer
3. Traning loop
    -forward pass: compute prediction and loss
    -backward pass: gradients
    -update weights
'''
class BertDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self,encodings):
        #super(BertDataset, self).__init__()
        self.encodings = encodings
    def __getitem__(self, index):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
    

class MLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.model_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
    def forward(self, input_ids=None,attention_mask=None,labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
    def get_optimizer(self,lr=None):
        if lr:
            return torch.optim.AdamW(self.parameters(),lr = lr)
        else:
            return torch.optim.AdamW(self.parameters(),lr = 1e-5)
    def save_model(self,savepath=None,early_stopping=False):
        flag='_es' if early_stopping else ''     
        #存在目录
        dir = os.join(savepath,time.strftime("%m-%d-%H-%m")+flag+".pt")
        
        if os.path.exists(os.path.dirname(savepath)):
            torch.save(self.state_dict(),dir)
        else:
            os.mkdir(os.path.dirname(savepath),777)
            torch.save(self.state_dict(),dir)
        print(f'save_dir:{dir}')
             
    def load_save_model(self,savepath):
        print(f"savepath: {savepath}")
        self.load_state_dict(torch.load(savepath)) #一般形式为model_dict=model.load_state_dict(torch.load(PATH))        
    def model_train(self,max_epoch:int,train_data,eval_data=None, lr=None,loss_fn=None,savepath=None,):
        min_val_loss=float('inf')
        optimizer = self.get_optimizer(lr=lr)
        lambda1 = lambda epoch: 0.65 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        size = len(train_data.dataset)
        eval_loop = tqdm(eval_data,leave=True)
        count=0
        for each in range(max_epoch):
            self.train()
            loop =  tqdm(train_data,leave=True)
            avg_loss = []
            for ind,batch in enumerate(loop):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self(input_ids,attention_mask=attention_mask,labels=labels)
                loss = outputs.loss
                #backward propogation
                loss.backward()
                optimizer.step()
                loop.set_description(f'Epoch {each}')
                loop.set_postfix(loss=loss.item(),lr=optimizer.param_groups[0]['lr'])
                avg_loss.append(loss.item())
                if ind %100 ==0:
                    current = ind*len(batch['input_ids'])
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]  avg_loss: {np.mean(avg_loss):>7f}")
            avg_loss = np.mean(avg_loss)
            #评估模式
            self.eval()
            with torch.no_grad():
                val_loss = []
                for ind,batch in enumerate(eval_loop):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self(input_ids,attention_mask=attention_mask,labels=labels)
                    loss = outputs.loss
                    val_loss.append(loss.item())
                    loop.set_description(f'eval: ')
                    loop.set_postfix(loss=loss.item())
                val_loss = np.mean(val_loss)
                
            if val_loss<min_val_loss:
                min_val_loss=val_loss
                count=0
            else:
                count=count+1
            if count >2:
                break
        if count>0:
            #早停
            print(f'earling_stopping,count={str(count)}')
            self.save_model(savepath,early_stopping=True)
        else:
            self.save_model(savepath)        
                    
        
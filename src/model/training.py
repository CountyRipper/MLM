import torch
from tqdm import tqdm
from model.MLM import BertDataset,MLM
from transformers import BertTokenizerFast
from utils.utils import read_texts,read_labels,Masked
from torch.utils.data import DataLoader

datadir = './dataset/Wiki10-31k/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_optimizer(model:torch.nn.Module,lr=1e-5):
    return torch.optim.AdamW(model.parameters(),lr = lr)


def get_dataloader(datadir,batch_size):
    print(f'datadir: {datadir}')
    print(f'batch_size={batch_size}')
    texts = read_texts(datadir=datadir)
    labels = read_labels(datadir=datadir)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    inputs = tokenizer(texts,return_tensors='pt',max_length=512,truncation=True,padding = 'max_length').to(device)
    #获取未被masked的
    print(len(inputs['input_ids']))
    inputs['labels'] = inputs.input_ids.detach().clone()

    for i in range(len(inputs['input_ids'])):
        inputs['input_ids'][i] = Masked(inputs['input_ids'][i])
    
    #inputs = list(map(lambda x:Masked(x),inputs))
    return DataLoader(BertDataset(inputs),batch_size=batch_size,shuffle=True)
 
def train(max_epoch:int,dataloader,model:MLM, optimizer, loss_fn=None):
    for each in range(max_epoch):
        loop =  tqdm(dataloader,leave=True)
        size = len(dataloader)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
            loss = outputs.loss
            #backward propogation
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch {each}')
            loop.set_postfix(loss=loss.item(),lr=optimizer.param_groups[0]['lr'])
            
            
        
    
from src.model.MLM import MLM
from src.model.training import *
import time
import os
if __name__ == '__main__':
    datadir = './dataset/Wiki10-31K/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MLM().to(device)
    train_texts = get_dataloader(datadir=datadir+"train_texts.txt",batch_size=8)
    eval_texts  = get_dataloader(datadir=datadir+"test_texts.txt",batch_size=8)
    model.model_train(max_epoch=10,train_data=train_texts,eval_data=eval_texts,lr=1e-5,savepath=os.path.join(datadir,'mlm'))
    
    
    
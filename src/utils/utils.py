from typing import List
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_labels(datadir)->List:
    print('datadir:', datadir)
    res = []
    with open(datadir,'r+') as r:
        res = [row.strip().split(", ") for row in r]
    return res

def read_texts(datadir)->List:
    print('datadir:',datadir)
    res=[]
    with open(datadir,'r') as f:
        for row in f:
            res.append(row.strip())        
    return res

'''
inputs is a dict including: input_ids, attention_mask, token_type_id and labels
'''
def Masked(inputs):
    rand = torch.rand(inputs.shape).to(device)
    # rand是一个跟input_ids相同形状的随机数组
    #0.15 is 15%
    # avoid padding, and end 
    mask_arr = (rand<0.15) *(inputs != 101)*(inputs !=102)*(inputs !=0)
    #获得 mask_arr 中的maske的索引位置的列表
    selection = []
    #print(mask_arr.shape)
    #shape[0] is the first dimension 第一维度
    tmp = torch.flatten(mask_arr.nonzero()).tolist()
    #print(len(tmp))
    #selection.append
            #取mask_arr第一维度的不为零的值形成列表
        # 103 is [mask]
    inputs[selection] = 103
    return inputs
    
    
    
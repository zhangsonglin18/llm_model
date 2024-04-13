# -*- coding: utf-8 -*-
import csv
import torch
import torch.utils.data as tud
from torch.nn.utils.rnn import pad_sequence
from tokenizer import Tokenizer

TRAIN_DATA_PATH = 'D:/extract/nlp-notebook/data_seq/train.tsv'
DEV_DATA_PATH = 'D:/extract/nlp-notebook/data_seq/dev.tsv'
MAX_LEN = 512
BATCH_SIZE = 4

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, token_type_ids_list, token_type_ids_for_mask_list, labels_list = [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        token_type_ids_for_mask_temp = instance["token_type_ids_for_mask"]
        labels_temp = instance["labels"]

        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        token_type_ids_for_mask_list.append(torch.tensor(token_type_ids_for_mask_temp, dtype=torch.long))
        labels_list.append(torch.tensor(labels_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "token_type_ids_for_mask": pad_sequence(token_type_ids_for_mask_list, batch_first=True, padding_value=-1),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100)}

class BertDataset(tud.Dataset):
    def __init__(self, data_path):
        super(BertDataset, self).__init__()

        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf, delimiter='\t')
            next(r)
            for row in r:
                summary = row[0]
                content = row[1]
                
                input_ids, token_type_ids, token_type_ids_for_mask, labels = Tokenizer.encode(content, summary, max_length=MAX_LEN)
                       
                self.data_set.append({"input_ids": input_ids, 
                                      "token_type_ids": token_type_ids, 
                                      "token_type_ids_for_mask": token_type_ids_for_mask,
                                      "labels": labels})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]
        
traindataset = BertDataset(TRAIN_DATA_PATH)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = BertDataset(DEV_DATA_PATH)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# for batch in valdataloader:
#     print(batch["input_ids"])
#     print(batch["input_ids"].shape)
#     print('------------------')
    
#     print(batch["token_type_ids"])
#     print(batch["token_type_ids"].shape)
#     print('------------------')
    
#     print(batch["token_type_ids_for_mask"])
#     print(batch["token_type_ids_for_mask"].shape)
#     print('------------------')
    
#     print(batch["labels"])
#     print(batch["labels"].shape)
#     print('------------------')
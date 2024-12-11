import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim

from model import DGASNet
from data_loader import DGAS_Dataset
import config as config

from utils import load_pkl, load_json
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(model, train_dataloader, optimizer, device):
    print("strat training")
    itr_start_time = time.time()


    for epoch in range(config.epochs):
        model.train()
        losses = []
        for index, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model(*batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if index % config.log_every == 0:
                elapsed = time.time() - itr_start_time
                info = 'itr:{} step_time:{} Loss={}'.format((index+1), elapsed/(index+1), np.mean(losses))
                print(info)

        if epoch+1 % 10==0:
            path = "model"+str(epoch)+".bin"
            save_model(model, path)
            print("*"*10+"save to "+path+"*"*10)



if __name__ == "__main__":
    set_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_desc_pkl = load_pkl("java_data/desc.pkl")
    train_apiseq_pkl = load_pkl("java_data/apiseq.pkl")

    api2doc_pkl = load_pkl("java_data/api2doc.pkl")
    vocab_desc_dict = load_json("java_data/vocab.desc.json")
    vocab_apiseq_dict = load_json("java_data/vocab.apiseq.json")
    vocab_apidoc_dict = load_json("java_data/vocab.apidoc.json")

    train_dataset = DGAS_Dataset(train_desc_pkl, train_apiseq_pkl, api2doc_pkl, config.max_len, vocab_desc_dict, vocab_apiseq_dict, vocab_desc_dict, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=1)

    model = DGASNet(config.emb_size, len(vocab_desc_dict), len(vocab_apidoc_dict), len(vocab_apidoc_dict), config.n_heads, config.c)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, eps=config.adam_epsilon)
    train(model, train_dataloader, optimizer, device)





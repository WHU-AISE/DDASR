import torch.nn as nn
from attention import MultiHeadAttention
from utils import PositionlEncoding
import torch
import torch.nn.functional as F

class DGASNet(nn.Module):
    def __init__(self, emb_size, text_vocab_size, api_vocab_size, doc_vocab_size, n_heads, c):
        super(DGASNet, self).__init__()
        self.emb_size = emb_size

        self.text_emb = nn.Embedding(num_embeddings=text_vocab_size, embedding_dim=emb_size, padding_idx=0)
        self.api_emb = nn.Embedding(num_embeddings=api_vocab_size, embedding_dim=emb_size, padding_idx=0)
        self.doc_emb = nn.Embedding(num_embeddings=doc_vocab_size, embedding_dim=emb_size, padding_idx=0)

        self.pos_enc = PositionlEncoding(emb_size)

        self.API_attention = MultiHeadAttention(n_heads, emb_size, emb_size, emb_size)
        self.Doc_attention = MultiHeadAttention(n_heads, emb_size, emb_size, emb_size)
        self.Text_attention = MultiHeadAttention(n_heads, emb_size, emb_size, emb_size)

        self.Doc_Guided_Text_to_Api_Attention = MultiHeadAttention(n_heads, emb_size, emb_size, emb_size)
        self.Doc_Text_Attention = MultiHeadAttention(n_heads, emb_size, emb_size, emb_size)

        self.c = c

    def enc_input(self, desc, api, doc):
        desc_emb = self.text_emb(desc)
        api_emb = self.api_emb(api)
        doc_emb = self.doc_emb(doc)
        return self.pos_enc(desc_emb), self.pos_enc(api_emb), self.pos_enc(doc_emb)

    def inner_modal_attention(self, desc_enc, api_enc, doc_enc):
        text_attention = self.Text_attention(desc_enc, desc_enc, desc_enc)
        api_attention = self.API_attention(api_enc, api_enc, api_enc)
        doc_attention = self.Doc_attention(doc_enc, doc_enc, doc_enc)
        
        return text_attention, api_attention, doc_attention
    
    def doc_guided_text_to_api_attention(self, desc_attention, api_attention, doc_attention):
        return self.Doc_Guided_Text_to_Api_Attention(desc_attention, doc_attention, api_attention)
    
    def text_to_doc_attention(self, desc_attention, api_attention, doc_attention):
        return self.Doc_Text_Attention(desc_attention, doc_attention, doc_attention)
    
    def doc_to_text_attention(self, desc_attention, api_attention, doc_attention):
        return self.Doc_Text_Attention(doc_attention, desc_attention, desc_attention)

    
    def fuse_API_attention(self, api_attention, doc_attention):
        return (api_attention+doc_attention)/2
            
    def cross_matching(self, desc,api,doc):
        desc_enc, api_enc, doc_enc = self.enc_input(desc, api, doc)
        desc_attention, api_attention, doc_attention = self.inner_modal_attention(desc_enc, api_enc, doc_enc)
        DGTA_attention = self.doc_guided_text_to_api_attention(desc_attention, api_attention, doc_attention)
        TD_attention = self.text_to_doc_attention(desc_attention, api_attention, doc_attention)
        DT_attention = self.doc_to_text_attention(desc_attention, api_attention, doc_attention)

        APIs_attention = torch.mean(self.fuse_API_attention(DGTA_attention, TD_attention),dim=1)
        words_attention = torch.mean(DT_attention, dim=1)

        return APIs_attention, words_attention
    
    def cal_sim(self, APIs_attention, words_attention):
        return F.cosine_similarity(APIs_attention, words_attention)

    
    



    def forward(self, pos_desc, pos_api, pos_doc, neg_api, neg_doc):
        pos_API, pos_Words = self.cross_matching(pos_desc, pos_api, pos_doc)
        neg_API, neg_Words = self.cross_matching(pos_desc, neg_api, neg_doc)

        pos_sim = self.cal_sim(pos_API, pos_Words)
        neg_sim = self.cal_sim(neg_API, neg_Words)

        loss = torch.relu(pos_sim-neg_sim+self.c).clamp(min=1e-6).mean()

        return loss
    

'''
if __name__ == '__main__':

    desc = torch.tensor([[1,2,3,4,5,6,0]])
    api = torch.tensor([[1,2,3,0,0,0,0]])
    doc =  torch.tensor([[1,2,3,4,5,6,7]])

    negdesc = torch.tensor([[1,1,1,1,2,3,0]])
    negapi = torch.tensor([[1,1,1,0,0,0,0]])
    negdoc =  torch.tensor([[1,1,1,1,1,1,1]])

    model = DGASNet(emb_size=10, text_vocab_size=20, api_vocab_size=10, doc_vocab_size=15, n_heads=6, c=0.1)
    loss = model(desc, api, doc, desc, api, doc)
    print(loss)
'''
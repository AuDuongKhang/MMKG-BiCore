from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from models import KBCModel



class QEB_wn(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(QEB_wn, self).__init__()    
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])


        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        
        self.scale = 0.1

        # Pre-trained embeddings
        wn_ling_f = r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f = r"../pre_train/matrix_wn_visual.npy"
        

        wn_ling, wn_visual = torch.tensor(np.load(wn_ling_f)), torch.tensor(np.load(wn_visual_f))
        self.img_vec = wn_visual.to(torch.float32)
        self.ling_vec = wn_ling.to(torch.float32)

        self.img_dimension = wn_visual.shape[-1]
        self.ling_dimension = wn_ling.shape[-1]
        self.modal_rank = rank 
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, self.modal_rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, self.modal_rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        self.tmp = nn.Parameter(torch.Tensor(rank, 6555), requires_grad=True)
        nn.init.xavier_uniform_(self.tmp)



    def forward(self, x, multi_modal):
        device = x.device
        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))

        # img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)
        # ling_embeddings = F.normalize(ling_embeddings, p=2, dim=-1)
        
        # print(img_embeddings.shape)

        fused_emb = torch.cat([img_embeddings, ling_embeddings], dim=-1) * self.scale
        # fused_emb = F.normalize(fused_emb, p=2, dim=-1) * self.scale

        # structure_embedding = F.normalize(self.embeddings[0].weight, p=2, dim=-1)
        structure_embedding = self.embeddings[0].weight * self.scale
        # embedding = (structure_embedding + fused_embeddings ) * self.scale

        lhs = structure_embedding[(x[:, 0])]
        rel = self.embeddings[1](x[:, 1])   
        rhs = structure_embedding[(x[:, 2])]

        lhs_mm = fused_emb[(x[:, 0])]
        rel_mm = self.embeddings[1](x[:, 1])   
        rhs_mm = fused_emb[(x[:, 2])]
        
        # complex
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        lhs_mm = lhs_mm[:, :self.rank], lhs_mm[:, self.rank:]
        rel_mm = rel_mm[:, :self.rank], rel_mm[:, self.rank:]
        rhs_mm = rhs_mm[:, :self.rank], rhs_mm[:, self.rank:]


        to_score = structure_embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]  

        to_score_mm = fused_emb
        to_score_mm = to_score_mm[:, :self.rank], to_score_mm[:, self.rank:]  
              
        score = ( 
            ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + 
             (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)) +  # (h, r, t) 

            ((lhs_mm[0] * rel_mm[0] - lhs_mm[1] * rel_mm[1]) @ to_score_mm[0].transpose(0, 1) + # (hm, rm, tm)
             (lhs_mm[0] * rel_mm[1] + lhs_mm[1] * rel_mm[0]) @ to_score_mm[1].transpose(0, 1)) + 
            
            ((lhs_mm[0] * rel[0] - lhs_mm[1] * rel[1]) @ to_score_mm[0].transpose(0, 1) + # (hm, rs, tm)
             (lhs_mm[0] * rel[1] + lhs_mm[1] * rel[0]) @ to_score_mm[1].transpose(0, 1)) +
            
            ((lhs_mm[0] * rel[0] - lhs_mm[1] * rel[1]) @ to_score[0].transpose(0, 1) + # (hm, rs, ts)
             (lhs_mm[0] * rel[1] + lhs_mm[1] * rel[0]) @ to_score[1].transpose(0, 1)) +
                
            ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_mm[0].transpose(0, 1) + # (hs, rs, tm)
             (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_mm[1].transpose(0, 1)) +
             
            ((lhs[0] * rel_mm[0] - lhs[1] * rel_mm[1]) @ to_score[0].transpose(0, 1) + # (hs, rm, ts)
             (lhs[0] * rel_mm[1] + lhs[1] * rel_mm[0]) @ to_score[1].transpose(0, 1)) +
             
            ((lhs[0] * rel_mm[0] - lhs[1] * rel_mm[1]) @ to_score_mm[0].transpose(0, 1) + # (hs, rm, tm)
             (lhs[0] * rel_mm[1] + lhs[1] * rel_mm[0]) @ to_score_mm[1].transpose(0, 1)) +
             
            ((lhs_mm[0] * rel_mm[0] - lhs_mm[1] * rel_mm[1]) @ to_score[0].transpose(0, 1) + # (hm, rm, ts)
             (lhs_mm[0] * rel_mm[1] + lhs_mm[1] * rel_mm[0]) @ to_score[1].transpose(0, 1)) 
        )

        factor =( 
            torch.sqrt(lhs[0] ** 2 + lhs_mm[0] ** 2 + lhs[1] ** 2 + lhs_mm[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel_mm[0] ** 2 + rel[1] ** 2 + rel_mm[1] ** 2 ),
            torch.sqrt(rhs[0] ** 2 + rhs_mm[0] ** 2 + rhs[1] ** 2 + rhs_mm[1] ** 2),   
        )

        return score, factor




class QEB_fb(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(QEB_fb, self).__init__()    
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])


        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        
        self.scale = 0.1

        # Pre-trained embeddings
        fb_ling_f = r"../pre_train/matrix_fb_ling.npy"
        fb_visual_f = r"../pre_train/matrix_fb_visual.npy"
        

        fb_ling, fb_visual = torch.tensor(np.load(fb_ling_f)), torch.tensor(np.load(fb_visual_f))
        self.img_vec = fb_visual.to(torch.float32)
        self.ling_vec = fb_ling.to(torch.float32)

        self.img_dimension = fb_visual.shape[-1]
        self.ling_dimension = fb_ling.shape[-1]
        self.modal_rank = rank 
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, self.modal_rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, self.modal_rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)


    def forward(self, x, multi_modal):
        device = x.device
        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))

        # img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)
        # ling_embeddings = F.normalize(ling_embeddings, p=2, dim=-1)
        
        # print(img_embeddings.shape)

        fused_emb = torch.cat([img_embeddings, ling_embeddings], dim=-1) * self.scale
        # fused_emb = F.normalize(fused_emb, p=2, dim=-1) * self.scale

        # structure_embedding = F.normalize(self.embeddings[0].weight, p=2, dim=-1)
        structure_embedding = self.embeddings[0].weight * self.scale
        # embedding = (structure_embedding + fused_embeddings ) * self.scale

        lhs = structure_embedding[(x[:, 0])]
        rel = self.embeddings[1](x[:, 1])   
        rhs = structure_embedding[(x[:, 2])]

        lhs_mm = fused_emb[(x[:, 0])]
        rel_mm = self.embeddings[1](x[:, 1])   
        rhs_mm = fused_emb[(x[:, 2])]
        
        # complex
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        lhs_mm = lhs_mm[:, :self.rank], lhs_mm[:, self.rank:]
        rel_mm = rel_mm[:, :self.rank], rel_mm[:, self.rank:]
        rhs_mm = rhs_mm[:, :self.rank], rhs_mm[:, self.rank:]


        to_score = structure_embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]  

        to_score_mm = fused_emb
        to_score_mm = to_score_mm[:, :self.rank], to_score_mm[:, self.rank:]  
              
        score = ( 
            ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + 
             (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)) +  # (h, r, t) 

            ((lhs_mm[0] * rel_mm[0] - lhs_mm[1] * rel_mm[1]) @ to_score_mm[0].transpose(0, 1) + # (hm, rm, tm)
             (lhs_mm[0] * rel_mm[1] + lhs_mm[1] * rel_mm[0]) @ to_score_mm[1].transpose(0, 1)) + 
            
            ((lhs_mm[0] * rel[0] - lhs_mm[1] * rel[1]) @ to_score_mm[0].transpose(0, 1) + # (hm, rs, tm)
             (lhs_mm[0] * rel[1] + lhs_mm[1] * rel[0]) @ to_score_mm[1].transpose(0, 1)) +
            
            ((lhs_mm[0] * rel[0] - lhs_mm[1] * rel[1]) @ to_score[0].transpose(0, 1) + # (hm, rs, ts)
             (lhs_mm[0] * rel[1] + lhs_mm[1] * rel[0]) @ to_score[1].transpose(0, 1)) +
                
            ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_mm[0].transpose(0, 1) + # (hs, rs, tm)
             (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_mm[1].transpose(0, 1)) +
             
            ((lhs[0] * rel_mm[0] - lhs[1] * rel_mm[1]) @ to_score[0].transpose(0, 1) + # (hs, rm, ts)
             (lhs[0] * rel_mm[1] + lhs[1] * rel_mm[0]) @ to_score[1].transpose(0, 1)) +
             
            ((lhs[0] * rel_mm[0] - lhs[1] * rel_mm[1]) @ to_score_mm[0].transpose(0, 1) + # (hs, rm, tm)
             (lhs[0] * rel_mm[1] + lhs[1] * rel_mm[0]) @ to_score_mm[1].transpose(0, 1)) +
             
            ((lhs_mm[0] * rel_mm[0] - lhs_mm[1] * rel_mm[1]) @ to_score[0].transpose(0, 1) + # (hm, rm, ts)
             (lhs_mm[0] * rel_mm[1] + lhs_mm[1] * rel_mm[0]) @ to_score[1].transpose(0, 1)) 
        )

        factor =( 
            torch.sqrt(lhs[0] ** 2 + lhs_mm[0] ** 2 + lhs[1] ** 2 + lhs_mm[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel_mm[0] ** 2 + rel[1] ** 2 + rel_mm[1] ** 2 ),
            torch.sqrt(rhs[0] ** 2 + rhs_mm[0] ** 2 + rhs[1] ** 2 + rhs_mm[1] ** 2),   
        )

        return score, factor

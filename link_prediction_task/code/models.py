from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import math

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        fb_ling_f=r'../pre_train/matrix_fb_ling.npy'
        fb_visual_f=r'../pre_train/matrix_fb_visual.npy'
        wn_ling_f=r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f=r"../pre_train/matrix_wn_visual.npy"
        fb_ling,fb_visual,wn_ling,wn_visual=torch.tensor(np.load(fb_ling_f)),torch.tensor(np.load(fb_visual_f)),torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
        multimodal_embeddings=[wn_ling,wn_visual]
        multimodal_embeddings1=[fb_ling,fb_visual]
        
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries,multimodal_embeddings)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks




class model_wn(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_wn, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        # Pre-trained embeddings
        wn_ling_f = r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f = r"../pre_train/matrix_wn_visual.npy"
        
        wn_ling, wn_visual = torch.tensor(np.load(wn_ling_f)), torch.tensor(np.load(wn_visual_f))
        self.img_vec = wn_visual.to(torch.float32)
        self.ling_vec = wn_ling.to(torch.float32)

        self.img_dimension = wn_visual.shape[-1]
        self.ling_dimension = wn_ling.shape[-1]
        
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 2* rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 2* rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
 

        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size      
        

    def forward(self, x, multi_modal):
        device = x.device

        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))

        
        fused_embeddings = (img_embeddings + ling_embeddings) * self.scale
        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # Split embeddings into real and imaginary parts for scoring
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
      
       
        return ( 
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1]* rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )

    

#FBIMG    

class model_fb(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_fb, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.scale = 0.2

        # Pre-trained embeddings
        fb_ling_f = r"../pre_train/matrix_fb_ling.npy"
        fb_visual_f = r"../pre_train/matrix_fb_visual.npy"
        
        fb_ling, fb_visual = torch.tensor(np.load(fb_ling_f)), torch.tensor(np.load(fb_visual_f))
        self.img_vec = fb_visual.to(torch.float32)
        self.ling_vec = fb_ling.to(torch.float32)

        self.img_dimension = fb_visual.shape[-1]
        self.ling_dimension = fb_ling.shape[-1]
        
        # Projection matrices
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size      




    def forward(self, x, multi_modal):
        device = x.device

        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))

        
        fused_embeddings = (img_embeddings + ling_embeddings) * self.scale
        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # Split embeddings into real and imaginary parts for scoring
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
      
       
        return ( 
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1]* rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )






    #db15k
class model_db(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_db, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        db_ling_f = r"../pre_train/DB15K-textual.pth"
        db_visual_f = r"../pre_train/DB15K-visual.pth"
        db_numeric_f = r"../pre_train/DB15K-visual.pth"

        # db_visual = torch.load(db_visual_f)
        # print(db_visual.shape)
        db_ling, db_visual, db_numeric = torch.load(db_ling_f, weights_only=True), \
                                         torch.load(db_visual_f, weights_only=True), \
                                         torch.load(db_numeric_f, weights_only=True)
        
        self.img_vec = db_visual.to(torch.float32).to('cuda')
        self.ling_vec = db_ling.to(torch.float32).to('cuda')

        self.img_dimension = db_visual.shape[-1]
        self.ling_dimension = db_ling.shape[-1]
         
       
        # # # Projection matrices
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        # self.mats_numeric = nn.Parameter(torch.Tensor(self.numeric_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_numeric)

    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size      
        




    def forward(self, x, multi_modal):
        device = x.device
        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
    
        # numeric_embeddings = self.numeric_vec.to(device).mm(self.mats_numeric.to(device))

        fused_embeddings = (img_embeddings + ling_embeddings) * self.scale
        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # Split embeddings into real and imaginary parts for scoring
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
      
       
        return ( 
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1]* rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )



#mkgw

class model_mkgw(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_mkgw, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        # db_ling_f = r"../pre_train/DB15K-textual.pth"
        # db_visual_f = r"../pre_train/DB15K-visual.pth"
        # db_numeric_f = r"../pre_train/DB15K-visual.pth"

        # # db_visual = torch.load(db_visual_f)
        # # print(db_visual.shape)
        # db_ling, db_visual, db_numeric = torch.load(db_ling_f, weights_only=True), \
        #                                  torch.load(db_visual_f, weights_only=True), \
        #                                  torch.load(db_numeric_f, weights_only=True)
        
        # self.img_vec = db_visual.to(torch.float32).to('cuda')
        # self.ling_vec = db_ling.to(torch.float32).to('cuda')

        # self.img_dimension = db_visual.shape[-1]
        # self.ling_dimension = db_ling.shape[-1]
         
        # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        # self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

        # print(self.img_embeddings.shape)
        
        # self.dim_e = 2 * rank
        # self.img_proj = nn.Sequential(
        #     nn.Linear(self.img_dimension, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, self.dim_e)
        # )
        # self.text_proj = nn.Sequential(
        #     nn.Linear(self.ling_dimension, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, self.dim_e)
        # )


        # print(db_ling.shape)
        # print(db_visual.shape)

        # self.numeric_vec = db_numeric.to(torch.float32)


        # self.numeric_dimension = db_numeric.shape[-1]
        
        # # # Projection matrices
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        # self.mats_numeric = nn.Parameter(torch.Tensor(self.numeric_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_numeric)

    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size      
        




    def forward(self, x, multi_modal):
        device = x.device
        # img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        # ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
    
        # numeric_embeddings = self.numeric_vec.to(device).mm(self.mats_numeric.to(device))

        # self.weight_img, self.weight_ling = torch.softmax(torch.stack([self.alpha, self.gamma]), dim=0)
        # # fused_embeddings_weighted = (self.weight_img * img_embeddings) + (self.weight_ling * ling_embeddings)
        # gate = torch.sigmoid(self.weight_img * img_embeddings + self.weight_ling * ling_embeddings)
        # fused_embeddings = (img_embeddings + ling_embeddings) * self.scale
        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        # embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale
        embedding = self.embeddings[0].weight






#FB

    # def forward(self, x, multi_modal):
    #     device = x.device

    #     # Use only the structural embeddings (TransE framework)
    #     embedding = self.embeddings[0].weight.to(device)  # Structural embeddings only
        
    #     # Extract embeddings for lhs (head), rel (relation), and rhs (tail)
        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # Split embeddings into real and imaginary parts for scoring
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
      
       
        return ( 
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1]* rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
            # lhs, rel, rhs,
        )
    


class model_mkgy(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_mkgy, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        # db_ling_f = r"../pre_train/DB15K-textual.pth"
        # db_visual_f = r"../pre_train/DB15K-visual.pth"
        # db_numeric_f = r"../pre_train/DB15K-visual.pth"

        # # db_visual = torch.load(db_visual_f)
        # # print(db_visual.shape)
        # db_ling, db_visual, db_numeric = torch.load(db_ling_f, weights_only=True), \
        #                                  torch.load(db_visual_f, weights_only=True), \
        #                                  torch.load(db_numeric_f, weights_only=True)
        
        # self.img_vec = db_visual.to(torch.float32).to('cuda')
        # self.ling_vec = db_ling.to(torch.float32).to('cuda')

        # self.img_dimension = db_visual.shape[-1]
        # self.ling_dimension = db_ling.shape[-1]
         
        # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        # self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

        # print(self.img_embeddings.shape)
        
        # self.dim_e = 2 * rank
        # self.img_proj = nn.Sequential(
        #     nn.Linear(self.img_dimension, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, self.dim_e)
        # )
        # self.text_proj = nn.Sequential(
        #     nn.Linear(self.ling_dimension, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, self.dim_e)
        # )


        # print(db_ling.shape)
        # print(db_visual.shape)

        # self.numeric_vec = db_numeric.to(torch.float32)


        # self.numeric_dimension = db_numeric.shape[-1]
        
        # # # Projection matrices
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        # self.mats_numeric = nn.Parameter(torch.Tensor(self.numeric_dimension, 2 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_numeric)

    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size      
        




    def forward(self, x, multi_modal):
        device = x.device
        # img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        # ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
    
        # numeric_embeddings = self.numeric_vec.to(device).mm(self.mats_numeric.to(device))

        # self.weight_img, self.weight_ling = torch.softmax(torch.stack([self.alpha, self.gamma]), dim=0)
        # # fused_embeddings_weighted = (self.weight_img * img_embeddings) + (self.weight_ling * ling_embeddings)
        # gate = torch.sigmoid(self.weight_img * img_embeddings + self.weight_ling * ling_embeddings)
        # fused_embeddings = (img_embeddings + ling_embeddings) * self.scale
        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        # embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale
        embedding = self.embeddings[0].weight






#FB

    # def forward(self, x, multi_modal):
    #     device = x.device

    #     # Use only the structural embeddings (TransE framework)
    #     embedding = self.embeddings[0].weight.to(device)  # Structural embeddings only
        
    #     # Extract embeddings for lhs (head), rel (relation), and rhs (tail)
        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # Split embeddings into real and imaginary parts for scoring
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
      
       
        return ( 
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1]* rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
            # lhs, rel, rhs,
        )

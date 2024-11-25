from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import pickle
import torch.nn.functional as F
import numpy as np
from config import alpha,beta,random_gate,forget_gate,remember_rate,constant


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    if not constant:
                        r_embeddings, img_embeddings= self.get_rhs(c_begin, chunk_size)
                        h_r = self.get_queries(these_queries)
                        n = len(h_r)
                        scores_str = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

                        for i in range(n):
                            i_alpha = self.alpha[(these_queries[i, 1])]
                            single_score = h_r[[i], :] @ (
                                    (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0,
                                                                                                                      1)
                            scores_str = torch.cat((scores_str, single_score.detach()), 0)
                    else:
                        rhs = self.get_rhs(c_begin, chunk_size) # 2000, 10182
                        q = self.get_queries(these_queries)     # bsz, 2000
                        scores_str = q @ rhs                    # bsz, 10182

                    lhs_img = F.normalize(self.img_vec[these_queries[:,0]], p=2, dim=1) # bsz, 1000
                    rhs_img = F.normalize(self.img_vec, p=2, dim=1).transpose(0, 1)     # bsz, 10182
                    score_img = lhs_img @ rhs_img                                       # bsz, 500, 10182
                    # beta=0.95
                    if forget_gate:
                        scores = torch.zeros_like(score_img, device=score_img.device)   # bsz, 10182
                        for i in range(len(these_queries)):
                            mode = queries[i, -1].item()
                            if mode == 0:   #（T， T）
                                scores[i] = scores_str[i]
                            elif mode == 1: # (I, T)
                                scores[i] = beta*scores_str[i]
                            else:           # (I, I)
                                scores[i] = beta*scores_str[i] + (1-beta)*score_img[i]*self.rel_pd[these_queries[i, 1]]
                    else:
                        scores = beta * scores_str + (1 - beta) * score_img
                    targets = self.score(these_queries) # bsz, 1
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

def sc_wz_01(len,num_1):
    A=[1 for i in range(num_1)]
    B=[0 for i in range(len-num_1)]
    C=A+B
    np.random.shuffle(C)
    return np.array(C,dtype=np.float)

def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return [line[:-1] for line in lines]

class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            finetune: bool = False,
            img_info='data/analogy/img_vec_id_analogy_vit.pickle',
            sig_alpha='data/analogy/rel_MPR_SIG_vit.pickle',
            rel_pd='data/analogy/rel_MPR_PD_vit_mrp{}.pickle'
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.finetune = finetune
        self.analogy_entids = read_txt('data/analogy/analogy_ent_id')
        self.analogy_relids = read_txt('data/analogy/analogy_rel_id')

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        if not constant:
            self.alpha=torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha=torch.cat((self.alpha,self.alpha),dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        if not random_gate:
            self.rel_pd=torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate),'rb'))).cuda()
        else:
            tmp=pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd=torch.from_numpy(sc_wz_01(len(tmp),np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd=torch.cat((self.rel_pd,self.rel_pd),dim=0)
        # self.alpha[self.img_info['missed'], :] = 1

        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_mats)

    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img
        else:
            rel = self.r_embeddings[1](x[:, 1])

            lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
            for i in range(len(x)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs[i] = self.r_embeddings[0](x[i, 0])
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                elif mode == 1: # (I, T)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                else:           # (I, I)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
     
            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img,1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img,rhs_img, 1).unsqueeze(1)


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str=torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            for i in range(len(score_str)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    continue
                elif mode == 1: # (I, T)
                    continue
                else:           # (I, I)
                    score_str[i] = beta * score_str[i] + (1-beta) * score_img[i]
            return score_str


    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            h_r = torch.cat((lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            for i in range(n):
                i_alpha = self.alpha[(x[i, 1])]
                single_score = h_r[[i], :] @ (
                            (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0, 1)
                ans = torch.cat((ans, single_score.detach()), 0)

            return (ans), (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        else:
            if not self.finetune:
                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                
                rel = self.r_embeddings[1](x[:, 1])
                
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    else:           # (I, I)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                rel = rel[:, :self.rank], rel[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]

                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]

                return (
                            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
                    ), (
                        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2) 
                    )
            else:
                '''analogical reasoning
                    x : [e_h, e_t, q, a, r, mode]
                    1. triple classification (e_h, ?, e_t) -> r
                    2. link prediction (a, r, ?) -> a
                '''
                # 1. triple classification
                rel = self.get_relations()    # 2000, 382
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                for i in range(len(x)):
                    mode = x[i, -1].item()
                    if mode == 0:   # （T， T）
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    elif mode == 1: # (I, T)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    else:           # (I, I)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 1]) + self.alpha * img_embeddings[x[i, 1]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]
                
                q = torch.cat([
                        lhs[0] * rhs[0] - lhs[1] * rhs[1],
                        lhs[0] * rhs[1] + lhs[1] * rhs[0]], 1)  # bsz, 2000
                scores_r = q @ rel
                scores_r = scores_r.argmax(dim=-1)              # bsz, 1
                
                # 2. link prediction
                pred_rel = self.r_embeddings[1](scores_r)       # bsz, 2000
                a_lhs = torch.rand((len(x), 2*self.rank)).to(x.device)  # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        a_lhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                    else:           # (I, I)
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                a_lhs = a_lhs[:, :self.rank], a_lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                pred_rel = pred_rel[:, :self.rank], pred_rel[:, self.rank:]

                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]

                return (
                            (a_lhs[0] * pred_rel[0] - a_lhs[1] * pred_rel[1]) @ to_score[0].transpose(0, 1) +
                            (a_lhs[0] * pred_rel[1] + a_lhs[1] * pred_rel[0]) @ to_score[1].transpose(0, 1)
                    ), (
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                        torch.sqrt(pred_rel[0] ** 2 + pred_rel[1] ** 2),
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2) 
                    )
                                     
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ],img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
            
    def get_relations(self):
        return self.r_embeddings[1].weight.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[(queries[:, 1])] * img_embeddings[
                (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        else:
            rel = self.r_embeddings[1](queries[:, 1])
            lhs = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            for i in range(len(queries)):
                mode = queries[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs[i] = self.r_embeddings[0](queries[i, 0])
                elif mode == 1: # (I, T)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                else:           # (I, I)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)


class Analogy(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            finetune: bool = False,
            img_info='data/analogy/img_vec_id_analogy_vit.pickle',
            sig_alpha='data/analogy/rel_MPR_SIG_vit.pickle',
            rel_pd='data/analogy/rel_MPR_PD_vit_mrp{}.pickle'
    ):
        super(Analogy, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.finetune = finetune
        self.analogy_entids = read_txt('data/analogy/analogy_ent_id')
        self.analogy_relids = read_txt('data/analogy/analogy_rel_id')

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])  # entity and relation
        
        self.ent_embeddings = nn.Embedding(self.sizes[0], self.rank * 2)
        self.rel_embeddings = nn.Embedding(self.sizes[1], self.rank * 2)

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        self.ent_embeddings.weight.data *= init_size
        self.rel_embeddings.weight.data *= init_size
        
        if not constant:
            self.alpha=torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha=torch.cat((self.alpha,self.alpha),dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        if not random_gate:
            self.rel_pd=torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate),'rb'))).cuda()
        else:
            tmp=pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd=torch.from_numpy(sc_wz_01(len(tmp),np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd=torch.cat((self.rel_pd,self.rel_pd),dim=0)

        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.post_mats)
        
    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img
        else:
            rel = self.r_embeddings[1](x[:, 1])
            rel_rel = self.rel_embeddings(x[:, 1])

            lhs_ent, rhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)
            lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
            for i in range(len(x)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs_ent[i] = self.ent_embeddings(x[i, 0])
                    rhs_ent[i] = self.ent_embeddings(x[i, 2])
                    lhs[i] = self.r_embeddings[0](x[i, 0])
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                elif mode == 1: # (I, T)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs_ent[i] = self.ent_embeddings(x[i, 2])
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                else:           # (I, I)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
     
            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img,1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img,rhs_img, 1).unsqueeze(1)


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            
            score_str=torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            ) + torch.sum((lhs_ent * rel_rel) * rhs_ent, 1, keepdim=True)
            # beta = 0.95
            for i in range(len(score_str)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    continue
                elif mode == 1: # (I, T)
                    continue
                else:           # (I, I)
                    score_str[i] = beta * score_str[i] + (1-beta) * score_img[i]
            return score_str

    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            h_r = torch.cat((lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            for i in range(n):
                i_alpha = self.alpha[(x[i, 1])]
                single_score = h_r[[i], :] @ (
                            (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0, 1)
                ans = torch.cat((ans, single_score.detach()), 0)

            return (ans), (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        else:
            if not self.finetune:
                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                
                rel_rel = self.rel_embeddings(x[:, 1])
                rel = self.r_embeddings[1](x[:, 1])
                
                lhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device)
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        lhs_ent[i] = self.ent_embeddings(x[i, 0])
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    else:           # (I, I)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                rel = rel[:, :self.rank], rel[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]
                
                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]
                
                to_score_ent = self.ent_embeddings.weight

                return (
                            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1) +
                            (lhs_ent * rel_rel) @ to_score_ent.transpose(0, 1)
                    ), (
                        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
                        torch.sqrt(lhs_ent[0] ** 2 + rel_rel[1] ** 2),
                    )
            else:
                '''analogical reasoning
                    x : [e_h, e_t, q, a, r, mode]
                    1. triple classification (e_h, ?, e_t) -> r
                    2. link prediction (a, r, ?) -> a
                '''
                # 1. triple classification
                rel = self.get_relations()    # 2000, 382
                rel_rel = self.rel_embeddings.weight
                
                lhs_ent, rhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                for i in range(len(x)):
                    mode = x[i, -1].item()
                    if mode == 0:   # （T， T）
                        lhs_ent[i] = self.ent_embeddings(x[i, 0])
                        rhs_ent[i] = self.ent_embeddings(x[i, 1])
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    elif mode == 1: # (I, T)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs_ent[i] = self.ent_embeddings(x[i, 1])
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    else:           # (I, I)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 1]) + self.alpha * img_embeddings[x[i, 1]]
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 1]) + self.alpha * img_embeddings[x[i, 1]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]
                
                q = torch.cat([
                        lhs[0] * rhs[0] - lhs[1] * rhs[1],
                        lhs[0] * rhs[1] + lhs[1] * rhs[0]], 1)  + lhs_ent * rhs_ent
                scores_r = q @ rel
                scores_r = scores_r.argmax(dim=-1)              # bsz, 1
                
                # 2. link prediction
                pred_rel = self.r_embeddings[1](scores_r)       # bsz, 2000
                pred_rel_rel = self.rel_embeddings(scores_r)
                
                a_lhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device)
                a_lhs = torch.rand((len(x), 2*self.rank)).to(x.device)  # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        a_lhs_ent[i] = self.ent_embeddings(x[i, 2])
                        a_lhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        a_lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                    else:           # (I, I)
                        a_lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                a_lhs = a_lhs[:, :self.rank], a_lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                pred_rel = pred_rel[:, :self.rank], pred_rel[:, self.rank:]

                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]
                
                to_score_ent = self.ent_embeddings.weight

                return (
                            (a_lhs[0] * pred_rel[0] - a_lhs[1] * pred_rel[1]) @ to_score[0].transpose(0, 1) +
                            (a_lhs[0] * pred_rel[1] + a_lhs[1] * pred_rel[0]) @ to_score[1].transpose(0, 1) +
                            (a_lhs_ent * pred_rel_rel) @ to_score_ent.transpose(0, 1)
                    ), (
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                        torch.sqrt(pred_rel[0] ** 2 + pred_rel[1] ** 2),
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                        torch.sqrt(a_lhs_ent ** 2 + pred_rel_rel ** 2)
                    )
                                     
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ],img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
            
    def get_relations(self):
        return self.r_embeddings[1].weight.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[(queries[:, 1])] * img_embeddings[
                (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        else:
            rel = self.r_embeddings[1](queries[:, 1])
            rel_rel = self.rel_embeddings(queries[:, 1])
            
            lhs = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            lhs_ent = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            for i in range(len(queries)):
                mode = queries[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs_ent[i] = self.ent_embeddings(queries[i, 0])
                    lhs[i] = self.r_embeddings[0](queries[i, 0])
                elif mode == 1: # (I, T)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                else:           # (I, I)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1) + lhs_ent * rel_rel


class ComplexImaginaryTransformation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexImaginaryTransformation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation1 = nn.ReLU() 
        self.activation2 = nn.LeakyReLU()
        self.activation3 = nn.Tanh() 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.dropout(x)
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        return x
class FactorizationMachine(nn.Module):
    def __init__(self, k: int):
        super(FactorizationMachine, self).__init__()
        self.k = k
    
    def forward(self, x):
        sum_square = torch.pow(torch.sum(x, dim=1), 2)  # (batch_size, k)
        square_sum = torch.sum(torch.pow(x, 2), dim=1)  # (batch_size, k)
        interaction = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)
        return interaction
class CrossNet(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x_cross = torch.matmul(x0, x.transpose(0, 1)) + self.cross_layers[i](x)
            x = x + x_cross
        return x

class GatedMechanism(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GatedMechanism, self).__init__()
        self.gate = nn.Linear(input_dim, output_dim) 
        self.sigmoid = nn.Sigmoid()
        self.expand = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        gate_value = self.sigmoid(self.gate(x1))
        if x1.shape != gate_value.shape:
            x1 = self.expand(x1) 
            
        if x2.shape != gate_value.shape:
            x2 = self.expand(x2) 

        return gate_value * x1 + (1 - gate_value) * x2


class BiComplex(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            finetune: bool = False,
            img_info='data/analogy/img_vec_id_analogy_vit.pickle',
            sig_alpha='data/analogy/rel_MPR_SIG_vit.pickle',
            rel_pd='data/analogy/rel_MPR_PD_vit_mrp{}.pickle',
            dropout_rate: float = 0.3,
            l2_lambda: float = 1e-3
    ):
        super(BiComplex, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.finetune = finetune
        self.analogy_entids = read_txt('data/analogy/analogy_ent_id')
        self.analogy_relids = read_txt('data/analogy/analogy_rel_id')

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fm_layer = FactorizationMachine(k=rank * 6)  # Factorization Machine k=rank*6
        self.cross_net = CrossNet(input_dim=rank * 4, num_layers=5)  
        self.layer_norm = nn.LayerNorm(self.rank * 2)  # Layer normalization
        self.gated_mechanism = GatedMechanism(input_dim=self.rank, output_dim=self.rank*2)  # Gated Mechanism for feature fusion

        self.l2_lambda = l2_lambda 
        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])  # entity and relation
        
        self.ent_embeddings = nn.Embedding(self.sizes[0], self.rank * 2)
        self.rel_embeddings = nn.Embedding(self.sizes[1], self.rank * 2)

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        self.ent_embeddings.weight.data *= init_size
        self.rel_embeddings.weight.data *= init_size
        
        if not constant:
            self.alpha = torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha = torch.cat((self.alpha, self.alpha), dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        
        # Image complex embedding
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        self.img_vec_real = torch.from_numpy(self.img_info).float().cuda()
        complex_imaginary_transform = ComplexImaginaryTransformation(self.img_dimension, 512, self.img_dimension).cuda()
        self.img_vec_imag = complex_imaginary_transform(self.img_vec_real)
        self.img_vec_complex = torch.stack((self.img_vec_real, self.img_vec_imag), dim=-1)
        
        if not random_gate:
            self.rel_pd = torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate), 'rb'))).cuda()
        else:
            tmp = pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd = torch.from_numpy(sc_wz_01(len(tmp), np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd = torch.cat((self.rel_pd, self.rel_pd), dim=0)
        
        self.post_mats_real = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank).uniform_(-0.1, 0.1), requires_grad=True)
        self.post_mats_imag = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank).uniform_(-0.1, 0.1), requires_grad=True)
        
        # Learnable beta for score combination
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # Attention layer to enhance combination between embeddings
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.rank * 2, num_heads=8) 
        
        # New Self-Attention mechanism for relations and entities
        self.self_attention = nn.MultiheadAttention(embed_dim=self.rank * 2, num_heads=4)  # Self-Attention Layer
        
        nn.init.xavier_uniform_(self.post_mats_real)
        nn.init.xavier_uniform_(self.post_mats_imag)

        # Warm-up Learning Rate Scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, total_steps=1000)
        self.clip_value = 1.0  # Gradient Clipping value

        
    def complex_mm(self, input_real, input_imag, weight_real, weight_imag):
        """
        Perform complex matrix multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        """
        output_real = torch.mm(input_real, weight_real) - torch.mm(input_imag, weight_imag)
        output_imag = torch.mm(input_real, weight_imag) + torch.mm(input_imag, weight_real)
        return output_real, output_imag

    def bicomplex_mm(self, input_real1, input_imag1, input_real2, input_imag2, weight_real1, weight_imag1, weight_real2, weight_imag2):
        """
        Perform bicomplex matrix multiplication: (a + bi + cj + dk) * (e + fi + gj + hk)
        """
        output_real1 = input_real1 @ weight_real1 - input_imag1 @ weight_imag1 - input_real2 @ weight_real2 + input_imag2 @ weight_imag2
        output_imag1 = input_real1 @ weight_imag1 + input_imag1 @ weight_real1 + input_real2 @ weight_imag2 + input_imag2 @ weight_real2

        output_real2 = input_real1 @ weight_real2 - input_imag1 @ weight_imag2 + input_real2 @ weight_real1 - input_imag2 @ weight_imag1
        output_imag2 = input_real1 @ weight_imag2 + input_imag1 @ weight_real2 - input_real2 @ weight_imag1 + input_imag2 @ weight_real1

        return output_real1, output_imag1, output_real2, output_imag2


    def score(self, x):
        # Calculate image embeddings using bicomplex_mm
        img_embeddings_real1, img_embeddings_imag1, img_embeddings_real2, img_embeddings_imag2 = self.bicomplex_mm(
            self.img_vec_complex[..., 0],
            self.img_vec_complex[..., 1],
            self.img_vec_complex[..., 2],
            self.img_vec_complex[..., 3],
            self.post_mats_real,
            self.post_mats_imag,
            self.post_mats_real,
            self.post_mats_imag
        )

        # Calculate lhs, rel, rhs for the given data
        alpha = self.alpha[(x[:, 1])].unsqueeze(-1)

        lhs_real = (1 - alpha) * self.r_embeddings[0](x[:, 0])[:, :self.rank] + alpha * img_embeddings_real1[(x[:, 0])]
        lhs_imag = (1 - alpha) * self.r_embeddings[0](x[:, 0])[:, self.rank:] + alpha * img_embeddings_imag1[(x[:, 0])]

        rel_real = self.r_embeddings[1](x[:, 1])[:, :self.rank]
        rel_imag = self.r_embeddings[1](x[:, 1])[:, self.rank:]

        rhs_real = (1 - alpha) * self.r_embeddings[0](x[:, 2])[:, :self.rank] + alpha * img_embeddings_real1[(x[:, 2])]
        rhs_imag = (1 - alpha) * self.r_embeddings[0](x[:, 2])[:, self.rank:] + alpha * img_embeddings_imag1[(x[:, 2])]

        # Apply Layer Normalization
        lhs_real = self.layer_norm(lhs_real)
        lhs_imag = self.layer_norm(lhs_imag)
        rel_real = self.layer_norm(rel_real)
        rel_imag = self.layer_norm(rel_imag)
        rhs_real = self.layer_norm(rhs_real)
        rhs_imag = self.layer_norm(rhs_imag)

        # Use Gated Mechanism to combine lhs, rel, rhs
        lhs_combined = self.gated_mechanism(lhs_real, lhs_imag)
        rhs_combined = self.gated_mechanism(rhs_real, rhs_imag)
        rel_combined = self.gated_mechanism(rel_real, rel_imag)

        # Self-Attention on combined embeddings
        combined_input = torch.cat([lhs_combined.unsqueeze(0), rel_combined.unsqueeze(0), rhs_combined.unsqueeze(0)], dim=0)
        attn_output, _ = self.self_attention(combined_input, combined_input, combined_input)
        attn_combined = attn_output.view(-1, attn_output.shape[2])

        # Apply 1D Convolutional layer to enrich the embeddings
        conv1d_layer = nn.Conv1d(in_channels=self.rank * 2, out_channels=self.rank * 4, kernel_size=1).cuda()
        enriched_lhs = conv1d_layer(lhs_combined.unsqueeze(0)).squeeze(0)
        enriched_rhs = conv1d_layer(rhs_combined.unsqueeze(0)).squeeze(0)
        enriched_rel = conv1d_layer(rel_combined.unsqueeze(0)).squeeze(0)

        # Use attention to enhance combination between lhs, rel, rhs
        query = enriched_lhs.unsqueeze(0)
        key = torch.cat([enriched_rhs.unsqueeze(0), enriched_rel.unsqueeze(0)], dim=0)
        value = torch.cat([enriched_rhs.unsqueeze(0), enriched_rel.unsqueeze(0)], dim=0)
        attn_output, _ = self.attention_layer(query, key, value)
        enhanced_lhs = attn_output.squeeze(0)

        # Combine real and imaginary components
        lhs = torch.cat([enhanced_lhs, lhs_imag], dim=-1)
        rel = torch.cat([rel_real, rel_imag], dim=-1)
        rhs = torch.cat([rhs_real, rhs_imag], dim=-1)

        # Calculate interaction using Factorization Machine layer
        combined_fm = torch.cat([lhs, rel, rhs], dim=-1)
        interaction_score_fm = self.fm_layer(combined_fm)

        # Cross Interaction using CrossNet
        combined_cross = torch.cat([lhs, rel, rhs], dim=-1)
        cross_interaction = self.cross_net(combined_cross)

        # Calculate scores using bicomplex
        score_str_real = (lhs_real * rel_real - lhs_imag * rel_imag) * rhs_real + (lhs_real * rel_imag + lhs_imag * rel_real) * rhs_imag
        score_str_imag = (lhs_real * rel_imag + lhs_imag * rel_real) * rhs_real - (lhs_imag * rel_imag - lhs_real * rel_real) * rhs_imag

        rel_pd = self.rel_pd[(x[:, 1])]
        lhs_img = self.img_vec_complex[(x[:, 0])]
        rhs_img = self.img_vec_complex[(x[:, 2])]

        score_img_real = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
        score_img_imag = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd

        # Use learnable beta for score combination
        beta = torch.sigmoid(self.beta_layer(lhs))  # Make beta more dynamic
        final_score_real = (
            beta * torch.sum(score_str_real, 1, keepdim=True)
            + (1 - beta) * score_img_real
            + interaction_score_fm
            + cross_interaction
        )
        final_score_imag = (
            beta * torch.sum(score_str_imag, 1, keepdim=True)
            + (1 - beta) * score_img_imag
            + interaction_score_fm
            + cross_interaction
        )

        return torch.cat((final_score_real, final_score_imag), dim=-1)



    def forward(self, x):
        # Perform complex matrix multiplication for image embeddings in bicomplex space
        img_embeddings_real, img_embeddings_imag = self.complex_mm(
            self.img_vec_complex[..., 0],  # Real part of img_vec
            self.img_vec_complex[..., 1],  # Imaginary part of img_vec
            self.post_mats_real,           # Real part of post_mats
            self.post_mats_imag            # Imaginary part of post_mats
        )

        if not constant:
            # Calculate lhs, rel, rhs in bicomplex space for the given mode
            # Ensure alpha has the correct dimensions for broadcasting
            alpha = self.alpha[(x[:, 1])].unsqueeze(-1)

            # Handle lhs embeddings (left-hand side entity embeddings)
            lhs_real = (1 - alpha) * self.r_embeddings[0](x[:, 0])[:, :self.rank] + alpha * img_embeddings_real[x[:, 0]]
            lhs_imag = (1 - alpha) * self.r_embeddings[0](x[:, 0])[:, self.rank:] + alpha * img_embeddings_imag[x[:, 0]]

            # Handle rel embeddings (relation embeddings)
            rel_real = self.r_embeddings[1](x[:, 1])[:, :self.rank]
            rel_imag = self.r_embeddings[1](x[:, 1])[:, self.rank:]

            # Handle rhs embeddings (right-hand side entity embeddings)
            rhs_real = (1 - alpha) * self.r_embeddings[0](x[:, 2])[:, :self.rank] + alpha * img_embeddings_real[x[:, 2]]
            rhs_imag = (1 - alpha) * self.r_embeddings[0](x[:, 2])[:, self.rank:] + alpha * img_embeddings_imag[x[:, 2]]
                  
            # Use dropout on embeddings to prevent overfitting
            lhs_real = self.dropout(lhs_real)
            lhs_imag = self.dropout(lhs_imag)
            rel_real = self.dropout(rel_real)
            rel_imag = self.dropout(rel_imag)
            rhs_real = self.dropout(rhs_real)
            rhs_imag = self.dropout(rhs_imag)

            # Define lhs, rel, and rhs in bicomplex space
            lhs = lhs_real, lhs_imag
            rel = rel_real, rel_imag
            rhs = rhs_real, rhs_imag
      

            # Calculate h_r in bicomplex space
            h_r_real = lhs[0] * rel[0] - lhs[1] * rel[1]
            h_r_imag = lhs[0] * rel[1] + lhs[1] * rel[0]
            h_r = torch.cat((h_r_real, h_r_imag), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            # Loop through the embeddings to compute scores
            for i in range(n):
                i_alpha = alpha[i]
                r_embedding_weight_real = self.r_embeddings[0].weight[:, :self.rank]
                r_embedding_weight_imag = self.r_embeddings[0].weight[:, self.rank:]

                # Calculate score for real and imaginary parts
                single_score_real = h_r_real[[i], :] @ (
                        (1 - i_alpha) * r_embedding_weight_real + i_alpha * img_embeddings_real).transpose(0, 1)
                single_score_imag = h_r_imag[[i], :] @ (
                        (1 - i_alpha) * r_embedding_weight_imag + i_alpha * img_embeddings_imag).transpose(0, 1)
                    
                # Concatenate real and imaginary scores
                single_score = torch.cat((single_score_real, single_score_imag), dim=-1)
                scores = self.dropout(scores)

                ans = torch.cat((ans, single_score), 0)  # Ensure no detaching to allow gradient computation
            
            # Regularization loss
            l2_norm = sum(param.pow(2.0).sum() for param in self.parameters())
            regularization_loss = self.l2_lambda * l2_norm

            return (ans + regularization_loss), (
                torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
            )

        else:
            if not self.finetune:
                alpha_unsqueezed = self.alpha.unsqueeze(-1)
                embedding_real = (1 - alpha_unsqueezed) * self.r_embeddings[0].weight[:, :self.rank] + alpha_unsqueezed * img_embeddings_real
                embedding_imag = (1 - alpha_unsqueezed) * self.r_embeddings[0].weight[:, self.rank:] + alpha_unsqueezed * img_embeddings_imag

                rel_real = self.r_embeddings[1](x[:, 1])[:, :self.rank]
                rel_imag = self.r_embeddings[1](x[:, 1])[:, self.rank:]

                # Adding dropout to embeddings to reduce overfitting
                embedding_real = self.dropout(embedding_real)
                embedding_imag = self.dropout(embedding_imag)
                rel_real = self.dropout(rel_real)
                rel_imag = self.dropout(rel_imag)

                # Initialize lhs and rhs as zeros to maintain consistent shapes
                lhs_real, lhs_imag = torch.zeros((len(x), self.rank)).to(x.device), torch.zeros((len(x), self.rank)).to(x.device)
                rhs_real, rhs_imag = torch.zeros((len(x), self.rank)).to(x.device), torch.zeros((len(x), self.rank)).to(x.device)

                # Loop through each example and set lhs and rhs based on mode
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    lhs_embedding = self.r_embeddings[0](x[i, 0])
                    rhs_embedding = self.r_embeddings[0](x[i, 2])

                    if mode == 0:  # (T, T)
                        lhs_real[i] = lhs_embedding[:self.rank]
                        lhs_imag[i] = lhs_embedding[self.rank:]
                        rhs_real[i] = rhs_embedding[:self.rank]
                        rhs_imag[i] = rhs_embedding[self.rank:]

                    elif mode == 1:  # (I, T)
                        lhs_real[i] = (1 - alpha_unsqueezed[i]) * lhs_embedding[:self.rank] + alpha_unsqueezed[i] * img_embeddings_real[x[i, 0]]
                        lhs_imag[i] = (1 - alpha_unsqueezed[i]) * lhs_embedding[self.rank:] + alpha_unsqueezed[i] * img_embeddings_imag[x[i, 0]]
                        rhs_real[i] = rhs_embedding[:self.rank]
                        rhs_imag[i] = rhs_embedding[self.rank:]

                    else:  # (I, I)
                        lhs_real[i] = (1 - alpha_unsqueezed[i]) * lhs_embedding[:self.rank] + alpha_unsqueezed[i] * img_embeddings_real[x[i, 0]]
                        lhs_imag[i] = (1 - alpha_unsqueezed[i]) * lhs_embedding[self.rank:] + alpha_unsqueezed[i] * img_embeddings_imag[x[i, 0]]
                        rhs_real[i] = (1 - alpha_unsqueezed[i]) * rhs_embedding[:self.rank] + alpha_unsqueezed[i] * img_embeddings_real[x[i, 2]]
                        rhs_imag[i] = (1 - alpha_unsqueezed[i]) * rhs_embedding[self.rank:] + alpha_unsqueezed[i] * img_embeddings_imag[x[i, 2]]

                # Adding dropout to lhs and rhs
                lhs_real = self.dropout(lhs_real)
                lhs_imag = self.dropout(lhs_imag)
                rhs_real = self.dropout(rhs_real)
                rhs_imag = self.dropout(rhs_imag)

                # Bicomplex multiplications and scores
                lhs = lhs_real, lhs_imag
                rel = rel_real, rel_imag
                rhs = rhs_real, rhs_imag

                to_score_real = embedding_real[:, :self.rank]
                to_score_imag = embedding_imag[:, self.rank:]

                scores = (
                    (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_real.transpose(0, 1) +
                    (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_imag.transpose(0, 1)
                )

                # Adding L2 regularization loss
                l2_norm = sum(param.pow(2.0).sum() for param in self.parameters())
                regularization_loss = self.l2_lambda * l2_norm

                # Adding regularization loss to scores
                final_scores = scores + regularization_loss

                # Return final scores and norms
                return (final_scores), (
                    torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                )

            else:
                # Handling finetune logic for analogical reasoning
                rel = self.get_relations()  # 2000, 382
                lhs_real, lhs_imag = torch.zeros((len(x), self.rank)).to(x.device), torch.zeros((len(x), self.rank)).to(x.device)
                rhs_real, rhs_imag = torch.zeros((len(x), self.rank)).to(x.device), torch.zeros((len(x), self.rank)).to(x.device)

                for i in range(len(x)):
                    mode = x[i, -1].item()
                    lhs_embedding = self.r_embeddings[0](x[i, 0])
                    rhs_embedding = self.r_embeddings[0](x[i, 2])

                    if mode == 0:  # (T, T)
                        lhs_real[i] = lhs_embedding[:self.rank]
                        lhs_imag[i] = lhs_embedding[self.rank:]
                        rhs_real[i] = rhs_embedding[:self.rank]
                        rhs_imag[i] = rhs_embedding[self.rank:]

                    elif mode == 1:  # (I, T)
                        if len(lhs_embedding.shape) == 1:
                            lhs_embedding = lhs_embedding.unsqueeze(0)  
                        if len(rhs_embedding.shape) == 1:
                            rhs_embedding = rhs_embedding.unsqueeze(0) 
                            
                        img_embedding_real_part = img_embeddings_real[x[i, 0]]
                        img_embedding_imag_part = img_embeddings_imag[x[i, 0]]
                        lhs_real[i] = (1 - self.alpha) * lhs_embedding[:, :self.rank] + self.alpha * img_embedding_real_part[:self.rank]
                        lhs_imag[i] = (1 - self.alpha) * lhs_embedding[:, self.rank:] + self.alpha * img_embedding_imag_part[self.rank:]
                        rhs_real[i] = rhs_embedding[:, :self.rank].squeeze(0)
                        rhs_imag[i] = rhs_embedding[:, self.rank:].squeeze(0)

                    else:  # (I, I)
                        if len(lhs_embedding.shape) == 1:
                            lhs_embedding = lhs_embedding.unsqueeze(0)        
                        if len(rhs_embedding.shape) == 1:
                            rhs_embedding = rhs_embedding.unsqueeze(0)                                                   
                        
                        img_embedding_real_part = img_embeddings_real[x[i, 0]]
                        img_embedding_imag_part = img_embeddings_imag[x[i, 0]]
                        lhs_real[i] = (1 - self.alpha) * lhs_embedding[:, :self.rank] + self.alpha * img_embedding_real_part[:self.rank]
                        lhs_imag[i] = (1 - self.alpha) * lhs_embedding[:, self.rank:] + self.alpha * img_embedding_imag_part[self.rank:]
                        rhs_real[i] = (1 - self.alpha) * rhs_embedding[:, :self.rank] + self.alpha * img_embedding_real_part[:self.rank]
                        rhs_imag[i] = (1 - self.alpha) * rhs_embedding[:, self.rank:] + self.alpha * img_embedding_imag_part[self.rank:]

                # Adding dropout to lhs and rhs
                lhs_real = self.dropout(lhs_real)
                lhs_imag = self.dropout(lhs_imag)
                rhs_real = self.dropout(rhs_real)
                rhs_imag = self.dropout(rhs_imag)
                
                lhs = lhs_real, lhs_imag
                rhs = rhs_real, rhs_imag

                q_real = lhs[0] * rhs[0] - lhs[1] * rhs[1]
                q_imag = lhs[0] * rhs[1] + lhs[1] * rhs[0]
                q = torch.cat([q_real, q_imag], 1)  # bsz, 2000
                scores_r = q @ rel
                scores_r = scores_r.argmax(dim=-1)  # bsz, 1

                # Link prediction logic
                pred_rel = self.r_embeddings[1](scores_r)  # bsz, 2000
                a_lhs_real, a_lhs_imag = torch.zeros((len(x), self.rank)).to(x.device), torch.zeros((len(x), self.rank)).to(x.device)

                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    a_lhs_embedding = self.r_embeddings[0](x[i, 2])
                    img_embedding_real_part = img_embeddings_real[x[i, 2]]
                    img_embedding_imag_part = img_embeddings_imag[x[i, 2]]
                        
                    if mode == 0:  # (T, T)
                        a_lhs_real[i] = a_lhs_embedding[:self.rank]
                        a_lhs_imag[i] = a_lhs_embedding[self.rank:]
                    elif mode == 1:  # (I, T)
                        if len(a_lhs_embedding.shape) == 1:
                            a_lhs_embedding = a_lhs_embedding.unsqueeze(0)
                        
                        a_lhs_real[i] = (1 - self.alpha) * a_lhs_embedding[:, :self.rank] + self.alpha * img_embedding_real_part[:self.rank]
                        a_lhs_imag[i] = (1 - self.alpha) * a_lhs_embedding[:, self.rank:] + self.alpha * img_embedding_imag_part[self.rank:]
                    else:  # (I, I)
                        if len(a_lhs_embedding.shape) == 1:
                            a_lhs_embedding = a_lhs_embedding.unsqueeze(0)
                        
                        a_lhs_real[i] = (1 - self.alpha) * a_lhs_embedding[:, :self.rank] + self.alpha * img_embedding_real_part[:self.rank]
                        a_lhs_imag[i] = (1 - self.alpha) * a_lhs_embedding[:, self.rank:] + self.alpha * img_embedding_imag_part[self.rank:]

                a_lhs = a_lhs_real, a_lhs_imag
                pred_rel_real = pred_rel[:, :self.rank]
                pred_rel_imag = pred_rel[:, self.rank:]

                embedding_real = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings_real
                embedding_imag = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings_imag
                to_score_real = embedding_real[:, :self.rank]
                to_score_imag = embedding_imag[:, self.rank:]

                # Adding L2 regularization loss
                l2_norm = sum(param.pow(2.0).sum() for param in self.parameters())
                regularization_loss = self.l2_lambda * l2_norm

                return (
                    (a_lhs[0] * pred_rel_real - a_lhs[1] * pred_rel_imag) @ to_score_real.transpose(0, 1) +
                    (a_lhs[0] * pred_rel_imag + a_lhs[1] * pred_rel_real) @ to_score_imag.transpose(0, 1) + regularization_loss
                ), (
                    torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                    torch.sqrt(pred_rel_real ** 2 + pred_rel_imag ** 2),
                    torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2)
                )
                                    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ],img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
            
    def get_relations(self):
        return self.r_embeddings[1].weight.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[(queries[:, 1])] * img_embeddings[
                (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        else:
            rel = self.r_embeddings[1](queries[:, 1])
            rel_rel = self.rel_embeddings(queries[:, 1])
            
            lhs = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            lhs_ent = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            for i in range(len(queries)):
                mode = queries[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs_ent[i] = self.ent_embeddings(queries[i, 0])
                    lhs[i] = self.r_embeddings[0](queries[i, 0])
                elif mode == 1: # (I, T)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                else:           # (I, I)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1) + lhs_ent * rel_rel

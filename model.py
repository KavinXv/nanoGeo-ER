from dataclasses import dataclass
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F

# 定义数据类 GeoConfig，用于存储超参数
@dataclass
class GeoConfig:
    lm_hidden: int = 768  # BERT 隐藏层大小
    c_em: int = 256        # 坐标嵌入大小
    n_em: int = 256       # 邻域嵌入大小
    a_em: int = 256        # 注意力嵌入大小


class NeighAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(NeighAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x_neigh_emb, x_distances):
        # 计算 Transformer 注意力
        attn_output, attn_weights = self.multihead_attn(x_neigh_emb, x_neigh_emb, x_neigh_emb)
        
        # 添加距离信息作为偏置项
        attn_weights = attn_weights - x_distances.unsqueeze(1)  # 让距离小的邻居权重大
        attn_weights = F.softmax(attn_weights, dim=-1)  # 归一化
        
        # 计算注意力分数并加权输出
        weighted_attn_output = torch.matmul(attn_weights, attn_output)  # (num_neighbors, embed_dim)
        
        # 计算最终邻域表示
        x_context = torch.sum(weighted_attn_output, dim=0)
        
        # 计算邻域最终向量
        x_context = torch.mean(x_context, dim=0)  # 变成 (embed_dim,)
        
        return x_context


class GeoER(nn.Module):
    def __init__(self, config, device='cpu', finetuning=True, dropout=0.2):
        super().__init__()
        self.hidden_size = config.lm_hidden
        self.device = device

        # 语言模型 (BERT)
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.neighbert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # **Transformer 多头注意力层（替代 self.attn）**
        # self.neigh_attn = nn.MultiheadAttention(embed_dim=config.a_em, num_heads=8)
        self.neigh_attn = NeighAttention(embed_dim=config.a_em, num_heads=8)

        # 线性层
        self.w_attn = nn.Linear(self.hidden_size, config.a_em)  # 降维
        self.n_attn = nn.Linear(2 * config.a_em, config.a_em)
        self.neigh_linear = nn.Linear(2 * config.a_em, config.n_em)
        self.coord_linear = nn.Linear(1, 2 * config.c_em)

        self.linear1 = nn.Linear(2 * self.hidden_size + 2 * config.c_em + config.n_em, (2 * self.hidden_size + 2 * config.c_em + config.n_em) // 2)
        self.linear2 = nn.Linear((2 * self.hidden_size + 2 * config.c_em + config.n_em) // 2, 2)

        # 激活函数
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_encode, x_coord, x_n, att_mask, neigh_mask, training=True):
        x, att_mask, neigh_mask, x_coord = x.to(self.device), att_mask.to(self.device), neigh_mask.to(self.device), x_coord.to(self.device)
        self.neighbert.eval()

        # 如果坐标的维度小于2，进行扩展
        while len(x_coord.shape) < 2:
            # 需要扩展,因为是单一数据，所以stack后就是一维的
            # 本来是[batch_size],升维之后是[1, batch_size],在后面才需要进行转置
            # print('coord.shape:',x_coord.shape)
            # x_coord = x_coord.unsqueeze(0)
            x_coord = x_coord.unsqueeze(1)

        # 处理文本
        if training:
            self.language_model.train()
            output = self.language_model(x, attention_mask=att_mask)
            n_output = self.neighbert(x_encode, attention_mask=neigh_mask)
        else:
            self.language_model.eval()
            with torch.no_grad():
                output = self.language_model(x, attention_mask=att_mask)
                n_output = self.neighbert(x_encode, attention_mask=neigh_mask)

        pooled_output = output[0][:, 0, :]
        n_pooled_output = n_output[0][:, 0, :]

        # 处理邻域信息
        x_neighbors = []

        for b in range(x.shape[0]):  # batch 维度
            x_neighborhood1, x_neighborhood2 = [], []
            x_distances1, x_distances2 = [], []

            with torch.no_grad():
                # 处理 name1 和 name2
                x_node1 = torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n[b]['name1'])['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], dim=1).squeeze()
                x_node2 = torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n[b]['name2'])['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], dim=1).squeeze()

                # 处理邻居1
                for x_n1 in x_n[b]['neigh1']:
                    x_neighborhood1.append(torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n1)['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], dim=1).squeeze())
                
                # 如果没有邻居，填充零向量
                if not len(x_neighborhood1):
                    x_neighborhood1.append(torch.zeros(self.hidden_size).to(self.device))
                
                # 处理邻居2
                for x_n2 in x_n[b]['neigh2']:
                    x_neighborhood2.append(torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n2)['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], dim=1).squeeze())
                
                # 如果没有邻居，填充零向量
                if not len(x_neighborhood2):
                    x_neighborhood2.append(torch.zeros(self.hidden_size).to(self.device))

                # 堆叠成张量
                x_neighborhood1 = torch.stack(x_neighborhood1).to(self.device)
                x_neighborhood2 = torch.stack(x_neighborhood2).to(self.device)

                # 将距离信息转换为Tensor
                x_distances1 = torch.tensor(x_distances1, dtype=torch.float).view(-1, 1).to(self.device)
                x_distances2 = torch.tensor(x_distances2, dtype=torch.float).view(-1, 1).to(self.device)

            # 拼接目标节点与邻居的嵌入向量
            x_concat1 = torch.cat([self.w_attn(x_node1).view(1,-1).repeat(x_neighborhood1.shape[0], 1), self.w_attn(x_neighborhood1)], 1)
            x_concat2 = torch.cat([self.w_attn(x_node2).view(1,-1).repeat(x_neighborhood2.shape[0], 1), self.w_attn(x_neighborhood2)], 1)
            # print('1', x_concat1.shape)
            # print('2', x_distances1.shape)

            # 继续进行 Transformer 操作和注意力计算
            x_neigh_emb1 = self.n_attn(x_concat1)  # (num_neighbors, a_em)
            x_neigh_emb2 = self.n_attn(x_concat2)

            # 计算 Transformer 注意力
            x_context1 = self.neigh_attn(x_neigh_emb1, x_distances1)
            x_context2 = self.neigh_attn(x_neigh_emb2, x_distances2)

            x_neighbors.append(self.relu(torch.cat([x_context1, x_context2])))

        # 邻域向量处理
        x_neighbors = torch.stack(x_neighbors)  # 现在，x_neighbors 中的张量都是相同大小的
        x_neighbors = self.neigh_linear(x_neighbors)

        # 坐标处理
        x_coord = self.coord_linear(x_coord)

        # 拼接 BERT 输出、邻域信息、坐标信息
        output = torch.cat([pooled_output, n_pooled_output, x_coord, x_neighbors], dim=1)

        # 分类
        output = self.linear2(self.drop(self.relu(self.linear1(output))))

        return F.log_softmax(output, dim=1)

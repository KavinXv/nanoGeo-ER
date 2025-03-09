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


class GeoER(nn.Module):
    def __init__(self, config: GeoConfig, device='cpu', finetuning=True, dropout=0.2):
        super().__init__()

        # 隐藏层大小
        self.hidden_size = config.lm_hidden

        # 加载预训练的BERT模型（用于处理文本数据）
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        # 加载另一个BERT模型，用于处理邻域数据
        self.neighbert = BertModel.from_pretrained('bert-base-uncased')
        # 加载BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 设备（CPU或GPU）
        self.device = device
        # 是否进行微调
        self.finetuning = finetuning

        # Dropout层，用于防止过拟合
        self.drop = nn.Dropout(dropout)
        # 注意力层
        self.attn = nn.Linear(self.hidden_size, 1)
        # 第一个全连接层，输入大小为文本特征 + 坐标嵌入 + 邻域嵌入
        self.linear1 = nn.Linear(self.hidden_size + 2 * config.c_em + config.n_em, (self.hidden_size + 2 * config.c_em + config.n_em) // 2)
        # 第二个全连接层，输出分类结果
        self.linear2 = nn.Linear((self.hidden_size + 2 * config.c_em + config.n_em) // 2, 2)

        # 邻域嵌入线性变换
        self.neigh_linear = nn.Linear(2 * config.a_em, config.n_em)
        # 坐标嵌入线性变换
        self.coord_linear = nn.Linear(1, 2 * config.c_em)

        # 邻域注意力的线性层
        # self.attn = nn.Linear(2 * config.a_em, 1)
        self.attn = nn.Linear(3 * config.a_em, 1)
        self.w_attn = nn.Linear(self.hidden_size, config.a_em)
        self.b_attn = nn.Linear(1, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU()

    def forward(self, x, x_coord, x_n, att_mask, training=True):
        # 将输入数据移动到指定设备（CPU或GPU）
        x = x.to(self.device)
        att_mask = att_mask.to(self.device)
        x_coord = x_coord.to(self.device)
        self.neighbert.eval()

        # 如果x的维度小于2，进行扩展
        if len(x.shape) < 2:
            # 无需扩展
            # print('x.shape:',x.shape)
            x = x.unsqueeze(0)

        # 如果attention mask的维度小于2，进行扩展
        if len(att_mask.shape) < 2:
            # 无需扩展
            # print('mask.shape:',att_mask.shape)
            att_mask = att_mask.unsqueeze(0)

        # 如果坐标的维度小于2，进行扩展
        while len(x_coord.shape) < 2:
            # 需要扩展,因为是单一数据，所以stack后就是一维的
            # 本来是[batch_size],升维之后是[1, batch_size],在后面才需要进行转置
            # print('coord.shape:',x_coord.shape)
            # x_coord = x_coord.unsqueeze(0)
            x_coord = x_coord.unsqueeze(1)

        b_s = x.shape[0]  # 获取批量大小

        # 如果处于训练模式且启用了微调
        if training and self.finetuning:
            self.language_model.train()
            self.train()
            # 将输入数据 x 和注意力掩码 att_mask 传递给语言模型（如BERT）。
            # x 是输入的文本序列，att_mask 是一个标记序列哪些部分应该被模型忽略（比如填充部分）
            output = self.language_model(x, attention_mask=att_mask)
            pooled_output = output[0][:, 0, :]  # 提取[CLS]标记的输出

        else:
            # 否则不微调，且不更新参数
            self.language_model.eval()
            with torch.no_grad():
                output = self.language_model(x, attention_mask=att_mask)
                pooled_output = output[0][:, 0, :]

        # 初始化邻域信息列表
        x_neighbors = []
        for b in range(b_s):
            # 初始化当前样本的邻域信息
            x_neighborhood1 = []
            x_neighborhood2 = []
            with torch.no_grad():
                # 对邻居实体进行BERT处理
                # self.tokenizer(x_n[b]['name1']) 会返回一个字典，
                # 其中包含了 input_ids 字段，这个字段是 name1 被分词并映射到 token ID 后的结果。
                # 这个结果是一个整数序列，代表了文本在词汇表中的索引。

                # self.neighbert(...) 返回一个包含两个部分的输出：第一个部分是一个形状为 [batch_size, seq_len, hidden_size] 的嵌入张量，
                # 第二个部分通常是 BERT 模型的池化输出或其他相关信息。这里 [0] 是选择第一个部分，
                # 即嵌入表示（不考虑第二部分的池化输出等）

                # 这里使用 torch.mean 对 BERT 的输出进行平均池化。
                # dim=1 表示沿着序列长度（seq_len）这一维进行求平均，
                # 也就是说，会将 BERT 输出的每个 token 的向量表示进行平均，从而得到一个固定长度的向量表示。
                # 假设BERT 输出的形状为 [1, seq_len, hidden_size]，通过 torch.mean(..., 1) 后，结果的形状将变为 [1, hidden_size]，即得到一个大小为 hidden_size（通常为 768）的向量。

                ### 就是为了将name转换成一个768维向量，与name长短无关（如：beijin、wulumuqi，处理后都只是一个768维向量）

                # .squeeze() 用来去除张量中形状为 1 的维度。在此例中，torch.mean(..., 1) 的输出形状是 [1, hidden_size]，
                # 而 .squeeze() 会去掉形状中的第一个维度，最终得到一个形状为 [hidden_size] 的一维张量，作为节点的嵌入表示。

                # 这两行代码的目的是通过 BERT 模型计算节点 name1 和 name2 的嵌入表示，最终得到两个形状为 [hidden_size]（通常为 768）的向量 x_node1 和 x_node2，表示每个节点的特征。
                x_node1 = torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n[b]['name1'])['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze()
                x_node2 = torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n[b]['name2'])['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze()
                # print(1)
                # print(x_node1.shape)

                # 处理邻居1的文本
                # 与处理name是一样的
                for x_n1 in x_n[b]['neigh1']:
                    x_neighborhood1.append(torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n1)['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze())

                # 如果邻居1为空，填充零向量
                if not len(x_neighborhood1):
                    x_neighborhood1.append(torch.zeros(768))

                # 处理邻居2的文本
                for x_n2 in x_n[b]['neigh2']:
                    x_neighborhood2.append(torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n2)['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze())

                # 如果邻居2为空，填充零向量
                if not len(x_neighborhood2):
                    x_neighborhood2.append(torch.zeros(768))

                # 将邻居的嵌入向量转换为Tensor
                # 把列表里的多个 768 维向量拼接成 张量，形成 (num_neighbors, 768) 的形状。
                x_neighborhood1 = torch.stack(x_neighborhood1).to(self.device)
                x_neighborhood2 = torch.stack(x_neighborhood2).to(self.device)
                # print(x_neighborhood1.shape)

                # 获取邻居的距离信息
                x_distances1 = x_n[b]['dist1']
                if not len(x_distances1):
                    x_distances1.append(1000)
                
                x_distances2 = x_n[b]['dist2']
                if not len(x_distances2):
                    x_distances2.append(1000)
                
                # 将距离信息转换为Tensor
                x_distances1 = torch.tensor(x_distances1, dtype=torch.float).view(-1, 1).to(self.device)
                x_distances2 = torch.tensor(x_distances2, dtype=torch.float).view(-1, 1).to(self.device)
                # print(x_distances1.shape)

            # 拼接目标节点与邻居的嵌入向量
            # 如果 x_node1 是形状 [768]，而 x_neighborhood1 有 5 个邻居，那么经过 view 和 repeat 后，
            # self.w_attn()会将768转到256
            # self.w_attn(x_node1).view(1,-1).repeat(5, 1) 会变成一个形状为 [5, 256] 的张量。

            # 拼接操作：self.w_attn(x_node1).view(1,-1).repeat(x_neighborhood1.shape[0], 1) 是一个形状为 [num_neighbors, 256] 的张量。
            # self.w_attn(x_neighborhood1) 也是一个形状为 [num_neighbors, 768] 变成 [num_neighbors, 256]的张量。
            # 所以，拼接之后的结果会是一个形状为 [num_neighbors, 256 + 256] = [num_neighbors, 512] 的张量。
            # x_concat1 = torch.cat([self.w_attn(x_node1).view(1,-1).repeat(x_neighborhood1.shape[0], 1), self.w_attn(x_neighborhood1)], 1)
            # x_concat2 = torch.cat([self.w_attn(x_node2).view(1,-1).repeat(x_neighborhood2.shape[0], 1), self.w_attn(x_neighborhood2)], 1)
            # print(x_concat1.shape)

            # 计算邻域的注意力得分
            # x_att1 = F.softmax(self.leaky(self.attn(x_concat1)) * self.b_attn(x_distances1),0)
            # x_att2 = F.softmax(self.leaky(self.attn(x_concat2)) * self.b_attn(x_distances2),0)

            x_att1 = F.softmax(self.leaky(self.attn(x_neighborhood1)) * self.b_attn(x_distances1),0)
            x_att2 = F.softmax(self.leaky(self.attn(x_neighborhood2)) * self.b_attn(x_distances2),0)


            # 计算上下文向量
            # 要开始求和了，在这里之前的张量形状可能与num_neighbors有关，这里之后就没有啦
            # self.w_attn(x_neighborhood1)*x_att1 -> [num_neighbors, 256] * [num_neighbors, 1] = [num_neighbors, 256]  
            # 相对应的部分相乘，每个邻居乘对应的注意力分数
            # sum之后就是[256]
            x_context1 = torch.sum(self.w_attn(x_neighborhood1)*x_att1,0) # 在dim = 0 上求和
            x_context2 = torch.sum(self.w_attn(x_neighborhood2)*x_att2,0)

            # 计算节点与邻域的相似度
            # x_context1 -> [256]
            # self.w_attn(x_node1) -> [256]
            # 对应的元素相乘
            # x_sim1 -> [256]
            x_sim1 = x_context1*self.w_attn(x_node1)
            x_sim2 = x_context2*self.w_attn(x_node2)

            # 将相似度拼接
            # 拼接之后 -> [512]
            x_neighbors.append(self.relu(torch.cat([x_sim1, x_sim2])))

        # 合并所有邻域信息
        x_neighbors = torch.stack(x_neighbors)
        x_neighbors = self.neigh_linear(x_neighbors)
        # x_neighbors -> [batch_size, 256]
        
        # 转置坐标并处理
        # print(x_coord.shape) # [1, batch_size]
        # x_coord = x_coord.transpose(0,1)
        # print(x_coord.shape) # [batch_size, 1]
        x_coord = self.coord_linear(x_coord)

        # 拼接BERT输出、坐标信息和邻域信息
        output = torch.cat([pooled_output, x_coord, x_neighbors], 1)
        # print(output.shape)

        # 通过全连接层进行分类，并应用激活函数
        output = self.linear2(self.drop(self.gelu(self.linear1(output))))
        
        # 使用log_softmax输出分类概率
        return F.log_softmax(output, dim=1)

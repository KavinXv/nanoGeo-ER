import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from math import radians, sin, cos, sqrt, atan2

# 提取实体的名称和地理坐标信息
def get_lat_long(entity):
    words = entity.lower().split()  # 将实体名称转为小写并分词
    name = None
    latitude = None
    longitude = None

    for i, word in enumerate(words):
        # 提取实体名称
        if i >= 2 and words[i-2] == 'name' and words[i-1] == 'val':
            name = ' '.join(words[i:])  # 从当前位置到末尾都是名称
        # 提取纬度
        elif i >= 2 and words[i-2] == 'latitude' and words[i-1] == 'val':
            try:
                latitude = float(word)
            except ValueError:
                pass
        # 提取经度
        elif i >= 2 and words[i-2] == 'longitude' and words[i-1] == 'val':
            try:
                longitude = float(word)
            except ValueError:
                pass

    # 清理名称字段，去掉多余的部分
    if name:
        name = name.split(' col ')[0].strip()

    return name, latitude, longitude


# 计算地理位置之间的距离，返回值归一化到[-1, 1]区间
def compute_dist(lat1, lon1, lat2, lon2, max_dist=100000):
    R = 6373.0  # 地球半径（单位：千米）

    # 检查输入的经纬度是否有效
    if None in [lat1, lon1, lat2, lon2]:
        return -1

    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # 计算经纬度差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # 使用Haversine公式计算两点之间的距离
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = R * c * 1000  # 转换为米
    # 将距离归一化到[-1, 1]区间
    dist = 2 * (dist / max_dist) - 1

    return dist


# 自定义 Dataset 类
class GeoDataset(Dataset):
    def __init__(self, path, n_path, max_seq_len=128, max_dist=2000):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 初始化BERT分词器
        self.max_seq_len = max_seq_len
        self.max_dist = max_dist
        self.data_x = []  # 存储文本数据（tokenized）
        self.coord_x = []  # 存储坐标数据（归一化后的距离）
        self.data_y = []  # 存储标签（0 或 1）
        self.neigh_x = []  # 存储邻域数据

        # 加载主数据
        with open(path, 'r', encoding = 'utf-8') as f:
            for line in f:
                arr = line.split('\t')  # 每行数据包含两个实体，通过tab分隔

                # 提取实体1和实体2的名称及其坐标
                e1, lat1, long1 = get_lat_long(arr[0])
                e2, lat2, long2 = get_lat_long(arr[1])

                if len(arr) > 2:  # 如果标签存在，处理文本和标签数据
                    # 使用BERT分词器将两个实体的名称拼接并tokenize
                    x = self.tokenizer.tokenize('[CLS] ' + e1 + ' [SEP] ' + e2 + ' [SEP]')
                    y = arr[2]  # 标签

                    # 如果token的长度小于最大长度，则进行填充
                    if len(x) < self.max_seq_len:
                        x = x + ['[PAD]'] * (self.max_seq_len - len(x))
                    else:
                        x = x[:self.max_seq_len]  # 如果token长度大于最大长度，则进行截断

                    # 将token转换为ID，并保存
                    self.data_x.append(self.tokenizer.convert_tokens_to_ids(x))
                    # 计算地理坐标的距离并保存
                    self.coord_x.append(compute_dist(lat1, long1, lat2, long2, self.max_dist))
                    self.data_y.append(int(y.strip()))  # 保存标签，转换为整数

        # 加载邻域数据（JSON格式）
        with open(n_path, encoding='utf-8') as f:
            self.neigh_x = json.load(f)  # 读取邻域数据

    def __len__(self):
        return len(self.data_x)  # 返回数据集的大小

    def __getitem__(self, idx):
        # 返回一个样本的所有数据
        return {
            'input_ids': torch.tensor(self.data_x[idx], dtype=torch.long),  # 文本数据
            'coords': torch.tensor(self.coord_x[idx], dtype=torch.float),  # 坐标数据
            'neighbors': self.neigh_x[idx],  # 邻域数据（保持 JSON 格式）
            'labels': torch.tensor(self.data_y[idx], dtype=torch.long)  # 标签
        }


# 自定义 collate_fn
def custom_collate_fn(batch):
    # 将批次数据中的每个字段分开
    input_ids = torch.stack([item['input_ids'] for item in batch])
    coords = torch.stack([item['coords'] for item in batch])
    neighbors = [item['neighbors'] for item in batch]  # 保持列表格式
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'coords': coords,
        'neighbors': neighbors,
        'labels': labels
    }

'''
# 测试
if __name__ == '__main__':
    # 数据集路径
    path = r'data\train_valid_test\osm_fsq\edi\test.txt'  # 主数据文件
    n_path = r'data\neighborhood_train_valid_test\osm_fsq\edi\n_test.json'  # 邻域数据文件

    # 创建数据集实例
    dataset = GeoDataset(path, n_path, max_dist=2000)  # max_dist 根据实际情况设置

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    # 测试 DataLoader
    for batch in dataloader:
        print(batch['input_ids'].shape)  # 文本数据的形状
        print(batch['coords'].shape)  # 坐标数据的形状
        print(batch['neighbors'])  # 邻域数据（JSON 格式）
        print(batch['labels'].shape)  # 标签的形状
        break
'''
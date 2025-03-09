## 1. Data

### 1.1 数据格式
数据部分为论文格式
```
名称1-维度1-经度1-地址1-邮编1   名称2-维度2-经度2-地址2-邮编2   1/0（是否属于同一实体）  
COL name VAL Grace North Edinburgh COL latitude VAL 55.96494943189891 COL longitude VAL -3.196916036169164 COL address VAL Logie Green Rd, 64 COL postalCode VAL EH7 4HQ 	COL name VAL Crombies of Edinburgh COL latitude VAL 55.95861906805779 COL longitude VAL -3.19005780781009 COL address VAL 97-101 Broughton St COL postalCode VAL EH1 3RZ 	0
```

### 1.2 数据集组成
- 三个数据源：OSM、FSQ、YELP
- 每个数据源中有四个城市：Singapore、Edinburgh、Toronto、Pittsburgh

- osm_fsq: OSM与FSQ两个数据集的实体关联数据
- osm_yelp: OSM与YELP两个数据集的实体关联数据

### 1.3 邻居数据集
- 格式是json格式
```
{
    "name1": "grace north edinburgh",
    "name2": "crombies of edinburgh",
    "neigh1": [],
    "neigh2": [
        "crombie's of edinburgh",
        "crest of edinburgh",
        "house of edinburgh",
        "macraes of edinburgh",
        "crombie's of edinburgh",
        "chic of edinburgh"
    ],
    "dist1": [],
    "dist2": [
        52,
        950,
        947,
        966,
        18,
        918
    ]
},
```

- 处理逻辑
在train_valid_test里面逐行读取数据，获取两个实体的信息，再与数据集里面所有的实体进行判断是否为邻居

## 2. 所需模块

### 2.1 function.py(数据集已经处理好，无需数据处理函数)
- 提取数据集数据函数：-> 获取每一行数据中(实体名称，维度，经度)
- 训练数据获取类：dataloader  ->  获取批次训练数据

### 2.2 model.py
- 二分类问题，给一个张量输出0/1
- @dataclass数据类: GeoConfig -> 存储超参数
- Geo-ER模型

模型逻辑：
1. 获取所需数据: 两个实体的文本数据(tokenized)、两个实体的距离数据、两个实体的邻居信息(未tokenized)
2. 转移到gpu
3. 扩展维度
4. 使用llm对两个实体进行特征提取
5. 处理两个实体的邻居信息：将实体与邻居拼接
6. 获取两个实体的邻居距离信息
7. 一些处理方法
8. 拼接信息：文本数据+距离信息+邻居信息
9. 全连接+softmax输出

### 2.3 train.py
- config 超参数设置
- 优化器设置
- 训练循环
- 测试函数
# nanoGeo-ER
# nanoGeo-ER

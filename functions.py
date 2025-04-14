import random
from difflib import SequenceMatcher
import math
from math import sin, cos, sqrt, atan2, radians
import json
import pandas as pd

def levenshtein(s1, s2):
    dp = [[0] * (len(s2)+1) for _ in range(len(s1)+1)]

    for i in range(len(s1)+1):
        for j in range(len(s2)+1):
            if i == 0:
                dp[i][j] = j  # 全部插入
            elif j == 0:
                dp[i][j] = i  # 全部删除
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],     # 删除
                                   dp[i][j-1],     # 插入
                                   dp[i-1][j-1])   # 替换
    edit_distance = dp[-1][-1]
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # 都是空字符串，认为完全相同
    return 1 - edit_distance / max_len

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

# 计算字符串的相似度，返回值在0到1之间，1表示完全相同
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
# 计算Jaccard相似度，两个列表的交集大小除以并集大小
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))  # 交集大小
    union = (len(list1) + len(list2)) - intersection  # 并集大小
    return float(intersection) / union

print(levenshtein("Hello world", "Hello word"))  # 输出 0.91
print(levenshtein("1234", "123456")) # 输出0.67
print(jaccard_similarity("Hello world", "Hello word")) # 输出0.61
print(jaccard_similarity("1234", "123456")) # 输出0.67
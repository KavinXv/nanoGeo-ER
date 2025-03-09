import logging
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model2 import GeoER, GeoConfig  # 导入GeoER模型和GeoConfig
from data import GeoDataset, custom_collate_fn  # 导入GeoDataset
import time

# 设置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# 超参数配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_seq_len = 128
max_dist = 2000
epochs = 2
batch_size = 32
lr = 3e-5
data_source = 'osm_fsq'
city = 'pit'
path = 'data/train_valid_test/'
n_path = 'data/neighborhood_train_valid_test/'
save_path = 'model_save/'  # 修改保存路径，使其可变
is_continue = 0

# 加载数据集
train_dataset = GeoDataset(path + data_source + '/' + city + '/train.txt',
                           n_path + data_source + '/' + city + '/n_train.json',
                           max_seq_len=max_seq_len, max_dist=max_dist)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

valid_dataset = GeoDataset(path + data_source + '/' + city + '/valid.txt',
                           n_path + data_source + '/' + city + '/n_valid.json',
                           max_seq_len=max_seq_len, max_dist=max_dist)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# 输出数据集加载的情况
print('Successfully loaded', city, '(', data_source, ') dataset')
print('Train size:', len(train_dataset))  # 打印训练集大小
print('Valid size:', len(valid_dataset))  # 打印验证集大小
# print('Test size:', len(test_dataset))  # 打印测试集大小


logger.info(f'Successfully loaded {city} ({data_source}) dataset')
logger.info(f'Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)}')

# 初始化模型
config = GeoConfig()
model = GeoER(config, device=device).to(device)

# 优化器与调度器
opt = optim.Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss()
num_steps = (len(train_dataset) // batch_size) * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)


# 断点续训：查找最新的模型文件
checkpoint_path = os.path.join(save_path, 'model_epoch_1_acc_0.9349.pt')
start_epoch = 0
if os.path.exists(checkpoint_path) and is_continue:
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        
        # 恢复模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复调度器状态
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复当前的 epoch
        start_epoch = checkpoint.get('epoch', 0)
        
        # 恢复最佳 F1 分数
        best_f1 = checkpoint.get('best_f1', 0.0)
        
        # 日志记录
        logger.info(f'Loaded saved model from {checkpoint_path} for resuming training from epoch {start_epoch + 1}.')
        logger.info(f'Best F1 score so far: {best_f1}')
    except Exception as e:
        logger.error(f'Failed to load checkpoint from {checkpoint_path}: {e}')
        raise
else:
    logger.info('No checkpoint found or resuming is disabled. Starting training from scratch.')
    start_epoch = 0
    best_f1 = 0.0
t1 = time.time()
train_steps = (len(train_dataset) + batch_size - 1) // batch_size
valid_steps = (len(valid_dataset) + batch_size - 1) // batch_size


for epoch in range(start_epoch, epochs):
    model.train()
    print('\n*** EPOCH:', epoch + 1, '***\n')
    logger.info(f'\n*** EPOCH: {epoch + 1} ***\n')
    
    t2 = time.time()
    for step, batch in enumerate(train_loader, 1):
        opt.zero_grad()
        x, x_coord, x_n, y = batch['input_ids'].to(device), batch['coords'].to(device), batch['neighbors'], batch['labels'].to(device)
        att_mask = (x != 0).int().to(device)
        
        pred = model(x, x_coord, x_n, att_mask)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        scheduler.step()
        t22 = time.time()
        dt2 = t22 - t2
        t2 = t22

        # 打印当前步的损失
        print('Step:', step, '/', train_steps, 'Loss:', loss.item(), 'Time:', dt2)
        logger.info(f'Step: {step}/{train_steps} | Loss: {loss.item()} | Time:{dt2}')
    t12 = time.time()
    dt1 = t12 - t1
    t1 = t12
    print('epoch:', epoch,'time:',dt1)

    # 验证阶段
    print('\n*** Validation Epoch:', epoch + 1, '***\n')
    model.eval()
    acc, prec, recall = 0.0, 0.0, 0.0
    total_samples = 0
    total_positives = 0
    total_pred_positives = 0

    t3 = time.time()
    valid_step = 1
    dt3 = 0
    for batch in valid_loader:
        y, x, x_coord, x_n = batch['labels'].view(-1).to(device), batch['input_ids'].to(device), batch['coords'].to(device), batch['neighbors']
        att_mask = (x != 0).int().to(device)
        
        pred = model(x, x_coord, x_n, att_mask, training=False)
        pred_labels = torch.argmax(pred, dim=1)

        print('Step:', valid_step, '/', valid_steps, 'Loss:', loss.item(), 'Time:', dt3)
        logger.info(f'Step: {valid_step}/{valid_steps} | Loss: {loss.item()} | Time:{dt3}')
        
        # 累加正确预测的样本数量
        acc += pred_labels.eq(y).sum().item()
        
        # 累加真正正类样本的数量
        recall += ((pred_labels == 1) & (y == 1)).sum().item()
        
        # 累加预测为正类的样本数量
        prec += (pred_labels == 1).sum().item()
        
        # 累加总样本数量和总正类样本数量
        total_samples += len(y)
        total_positives += (y == 1).sum().item()

        valid_step += 1
        t32 = time.time()
        dt3 = t32 - t3
        t3 = t32

    # 计算最终指标
    acc /= total_samples
    prec = recall / prec if prec > 0 else 0
    recall = recall / total_positives if total_positives > 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
    
    # 输出指标
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", recall)
    print("f1-Score:", f1)

    logger.info(f'Validation - Accuracy: {acc}, Precision: {prec}, Recall: {recall}, F1-Score: {f1}')
    
    # 保存模型，并包含epoch数和loss信息
    model_filename = f'model_epoch_{epoch + 1}_acc_{acc:.4f}.pt'
    checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss.item(),
    'f1': f1,
    'best_f1': best_f1
    }
    # torch.save(checkpoint, os.path.join(save_path, 'last_ckpt.pt'))
    torch.save(checkpoint, os.path.join(save_path, model_filename))
    print(f'Saved model as {model_filename}')
    logger.info(f'Saved model as {model_filename}')


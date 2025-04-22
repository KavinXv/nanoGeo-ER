import re
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    train_loss = []
    metrics = {
        'epoch': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    epoch = None  # 初始化 epoch

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 训练损失解析
            train_match = re.search(r"Epoch: (\d+) Step: (\d+)/\d+ \| Loss: ([\deE.-]+)", line)
            if train_match:
                epoch, step, loss_str = int(train_match.group(1)), int(train_match.group(2)), train_match.group(3)
                loss = float(loss_str)
                global_step = epoch * 79 + step  # 注意这里 79 是每个 epoch 的 step 数
                train_loss.append((global_step, loss))

            # 验证指标解析
            val_match = re.search(
                r"Validation - Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)", line
            )
            if val_match:
                if epoch is not None:
                    metrics['epoch'].append(epoch)
                    metrics['accuracy'].append(float(val_match.group(1)))
                    metrics['precision'].append(float(val_match.group(2)))
                    metrics['recall'].append(float(val_match.group(3)))
                    metrics['f1'].append(float(val_match.group(4)))
                else:
                    print(f"警告: 未能识别 epoch -> {line.strip()}")

    return train_loss, metrics

def plot_results(train_loss, metrics, loss_ylim=None, metric_ylim=None):
    plt.figure(figsize=(14, 6))

    # 子图1：训练损失
    plt.subplot(1, 2, 1)
    global_steps = [gs for gs, _ in train_loss]
    losses = [l for _, l in train_loss]
    plt.plot(global_steps, losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel('Global Step (Epoch * 79 + Step)')
    plt.ylabel('Loss')
    plt.yscale('log')
    if loss_ylim:
        plt.ylim(loss_ylim)
    plt.title('Training Loss (Log Scale)')
    plt.legend()

    # 子图2：评估指标
    plt.subplot(1, 2, 2)
    epochs = metrics['epoch']
    plt.plot(epochs, metrics['accuracy'], marker='o', label='Accuracy')
    plt.plot(epochs, metrics['precision'], marker='s', label='Precision')
    plt.plot(epochs, metrics['recall'], marker='^', label='Recall')
    plt.plot(epochs, metrics['f1'], marker='d', label='F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    if metric_ylim:
        plt.ylim(metric_ylim)
    plt.title('Validation Metrics over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 用法
log_file = './training.log'  # 替换为你的实际日志文件路径
log_file = './training_osm_yelp_pit.log' 
train_loss, metrics = parse_log_file(log_file)
plot_results(train_loss, metrics, loss_ylim=(0, 7), metric_ylim=(0.85, 1.0))


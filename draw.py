import re
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    train_loss = []
    accuracy = []
    epoch = None  # 确保 `epoch` 变量初始化

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析训练损失
            train_match = re.search(r"Epoch: (\d+) Step: (\d+)/\d+ \| Loss: ([\deE.-]+)", line)
            if train_match:
                epoch, step, loss_str = int(train_match.group(1)), int(train_match.group(2)), train_match.group(3)
                loss = float(loss_str)  # 直接转换，能自动识别 e-05 形式
                global_step = epoch * 79 + step  # 计算全局 step
                train_loss.append((global_step, loss))

            # 解析验证集 Accuracy
            val_match = re.search(r"Validation - Accuracy: ([\d.]+)", line)
            if val_match:
                if epoch is not None:  # 确保 `epoch` 不是 None
                    acc = float(val_match.group(1))
                    accuracy.append((epoch, acc))
                else:
                    print(f"警告: 解析到 Accuracy 但未找到对应 Epoch，跳过该条记录 -> {line.strip()}")

    return train_loss, accuracy

def plot_results(train_loss, accuracy):
    plt.figure(figsize=(12, 5))

    # 1. 训练损失 Loss vs. Global Step
    plt.subplot(1, 2, 1)
    global_steps = [gs for gs, _ in train_loss]
    losses = [l for _, l in train_loss]
    plt.plot(global_steps, losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel('Global Step (Epoch * 79 + Step)')
    plt.ylabel('Loss')
    plt.yscale('log')  # 适应 e-05 级别的 Loss
    plt.title('Training Loss vs Global Step (Log Scale)')
    plt.legend()

    # 2. 准确率 Accuracy vs. Epoch
    plt.subplot(1, 2, 2)
    epochs_acc = [e for e, _ in accuracy]
    acc_values = [a for _, a in accuracy]
    plt.plot(epochs_acc, acc_values, marker='s', linestyle='-', color='r', label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 运行代码
log_file = 'training.log'  # 请替换为你的日志文件路径
train_loss, accuracy = parse_log_file(log_file)
plot_results(train_loss, accuracy)

# plot_loss.py

import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(log_path='training_log_no_tensorboard.txt'):
    """
    解析NeRF训练日志文件，从中提取训练步数(step)和总损失(total_loss)。
    
    Args:
        log_path (str): 训练日志文件的路径。
        
    Returns:
        pandas.DataFrame: 一个包含 'step' 和 'total_loss' 两列的DataFrame，
                          如果没有找到数据则返回 None。
    """
    # 定义一个正则表达式来匹配日志中的关键信息。
    # - re.compile() 预编译正则表达式以提高效率。
    # - \s* 匹配任意数量的空白字符。
    # - (\d+) 捕获一个或多个数字（这是step）。
    # - .*? 非贪婪匹配任意字符，直到下一个模式。
    # - ([0-9.]+) 捕获一个或多个数字或小数点（这是loss）。
    log_pattern = re.compile(r'step:\s*(\d+).*?total_loss:\s*([0-9.]+)')
    
    # 用于存储解析出的数据
    data = []
    
    print(f"正在读取日志文件: {log_path}")
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 在每一行中搜索匹配的模式
                match = log_pattern.search(line)
                if match:
                    # 如果找到匹配，提取捕获的组
                    step = int(match.group(1))
                    total_loss = float(match.group(2))
                    data.append({'step': step, 'total_loss': total_loss})
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 '{log_path}'。请确保文件名正确并且文件在当前目录下。")
        return None

    if not data:
        print("警告: 未在日志文件中找到匹配的数据行。")
        print("请检查日志文件内容和正则表达式 'log_pattern' 是否匹配你的日志格式。")
        return None
        
    print(f"成功解析 {len(data)} 个数据点。")
    
    # 将数据列表转换为Pandas DataFrame，方便后续处理
    return pd.DataFrame(data)

def plot_loss_curve(df):
    """
    使用matplotlib从DataFrame绘制Loss曲线图。
    
    Args:
        df (pandas.DataFrame): 包含 'step' 和 'total_loss' 的DataFrame。
    """
    if df is None or df.empty:
        print("数据为空，无法绘制图表。")
        return

    # 创建一个新的图形窗口，并设置大小
    plt.figure(figsize=(15, 7))
    
    # 1. 绘制原始的、未经平滑的Loss数据点
    #    alpha=0.4 使其半透明，作为背景参考
    plt.plot(df['step'], df['total_loss'], label='Raw Total Loss', alpha=0.4, color='gray', linestyle='-')
    
    # 2. 计算并绘制平滑后的Loss曲线
    #    .rolling() 是Pandas提供的计算移动窗口统计量的强大功能
    #    window=100 表示每个点的值是它自己及前面99个点的平均值
    smoothing_window = 100 
    df['smoothed_loss'] = df['total_loss'].rolling(window=smoothing_window, min_periods=1).mean()
    plt.plot(df['step'], df['smoothed_loss'], label=f'Smoothed Loss (window={smoothing_window})', color='dodgerblue', linewidth=2.5)

    # --- 图表美化 ---
    plt.title('Training Loss Curve Analysis', fontsize=16)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=max(df['total_loss'].quantile(0.95), 0.5)) # 动态调整y轴上限，避免极端值影响
    plt.xlim(left=0) # x轴从0开始
    
    # 保存图表到文件
    output_path = 'loss_curve_from_log.png'
    try:
        plt.savefig(output_path, dpi=300) # dpi=300 保存为高分辨率图片
        print(f"Loss曲线图已保存至: {output_path}")
    except Exception as e:
        print(f"保存图片失败: {e}")
    
    # 在窗口中显示图表
    plt.show()


if __name__ == '__main__':
    # 定义日志文件的路径
    LOG_FILE_PATH = 'log.txt'
    
    # 步骤1: 解析日志文件，得到DataFrame
    log_dataframe = parse_log_file(LOG_FILE_PATH)
    
    # 步骤2: 绘制并展示Loss曲线
    plot_loss_curve(log_dataframe)
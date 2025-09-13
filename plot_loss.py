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

    log_pattern = re.compile(r'step:\s*(\d+).*?total_loss:\s*([0-9.]+)')
    
    data = []
    
    print(f"正在读取日志文件: {log_path}")
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
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

    plt.figure(figsize=(15, 7))
    
    plt.plot(df['step'], df['total_loss'], label='Raw Total Loss', alpha=0.4, color='gray', linestyle='-')
    
    smoothing_window = 100 
    df['smoothed_loss'] = df['total_loss'].rolling(window=smoothing_window, min_periods=1).mean()
    plt.plot(df['step'], df['smoothed_loss'], label=f'Smoothed Loss (window={smoothing_window})', color='dodgerblue', linewidth=2.5)

    plt.title('Training Loss Curve Analysis', fontsize=16)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=max(df['total_loss'].quantile(0.95), 0.5)) 
    plt.xlim(left=0) 
    
    output_path = 'loss_curve_from_log.png'
    try:
        plt.savefig(output_path, dpi=300) 
        print(f"Loss曲线图已保存至: {output_path}")
    except Exception as e:
        print(f"保存图片失败: {e}")
    
    plt.show()


if __name__ == '__main__':
    LOG_FILE_PATH = 'log.txt'
    
    log_dataframe = parse_log_file(LOG_FILE_PATH)
    
    plot_loss_curve(log_dataframe)
# plot_loss.py

import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(log_path='log.txt'):
    """
    解析NeRF训练日志文件，从中提取训练的loss以及每个epoch结束时的验证指标。
    """
    # 匹配训练日志行，提取 step 和 total_loss
    train_pattern = re.compile(r'step:\s*(\d+).*?total_loss:\s*([0-9.]+)')
    # 匹配验证结果行，提取 PSNR
    psnr_pattern = re.compile(r'Average PSNR:\s*([0-9.]+)')
    # 匹配验证结果行，提取 SSIM
    ssim_pattern = re.compile(r'Average SSIM:\s*([0-9.]+)')
    # 匹配epoch结束的标志，以获取当前step
    epoch_end_step_pattern = re.compile(r'epoch:\s*(\d+)\s*step:\s*(\d+)')

    train_data = []
    eval_data = []
    current_step = 0

    print(f"正在读取日志文件: {log_path}")
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 尝试匹配训练日志
                train_match = train_pattern.search(line)
                if train_match:
                    step = int(train_match.group(1))
                    total_loss = float(train_match.group(2))
                    train_data.append({'step': step, 'total_loss': total_loss})
                    current_step = step # 持续更新当前step
                    continue # 处理完这行就跳到下一行

                # 尝试匹配PSNR日志
                psnr_match = psnr_pattern.search(line)
                if psnr_match:
                    psnr = float(psnr_match.group(1))
                    # 将PSNR与它所属的那个epoch结束时的step关联起来
                    eval_data.append({'step': current_step, 'psnr': psnr})
                    continue

                # 尝试匹配SSIM日志
                ssim_match = ssim_pattern.search(line)
                if ssim_match:
                    ssim = float(ssim_match.group(1))
                    # 找到最近的一个带psnr的记录，并把ssim加进去
                    if eval_data and 'ssim' not in eval_data[-1]:
                        eval_data[-1]['ssim'] = ssim
                    else: # 如果找不到，就创建一个新记录
                        eval_data.append({'step': current_step, 'ssim': ssim})
    
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 '{log_path}'。")
        return None, None

    if not train_data and not eval_data:
        print("警告: 未在日志文件中找到任何可解析的数据。")
        return None, None
        
    print(f"成功解析 {len(train_data)} 个训练数据点和 {len(eval_data)} 个评估数据点。")
    
    # 将数据列表转换为Pandas DataFrame
    train_df = pd.DataFrame(train_data)
    eval_df = pd.DataFrame(eval_data)
    
    return train_df, eval_df

def plot_metrics(train_df, eval_df):
    """
    使用matplotlib从DataFrame绘制Loss, PSNR, SSIM三条曲线。
    """
    if train_df is None or train_df.empty:
        print("训练数据为空，无法绘制图表。")
        return

    # plt.subplots() 可以方便地创建网格状的子图
    # nrows=3, ncols=1 表示3行1列
    # sharex=True 表示所有子图共享同一个X轴（Training Step）
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True)
    fig.suptitle('NeRF Training Process Analysis', fontsize=20)

    # 在第一个子图 (axes[0]) 上绘制 Loss 曲线 
    ax1 = axes[0]
    ax1.plot(train_df['step'], train_df['total_loss'], label='Raw Total Loss', alpha=0.3, color='gray')
    smoothing_window = 100
    train_df['smoothed_loss'] = train_df['total_loss'].rolling(window=smoothing_window, min_periods=1).mean()
    ax1.plot(train_df['step'], train_df['smoothed_loss'], label=f'Smoothed Loss (window={smoothing_window})', color='dodgerblue', linewidth=2)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training Loss over Steps', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--')
    ax1.set_ylim(bottom=0)

    # 3. 在第二个子图 (axes[1]) 上绘制 PSNR 曲线 
    if eval_df is not None and not eval_df.empty and 'psnr' in eval_df.columns:
        ax2 = axes[1]
        ax2.plot(eval_df['step'], eval_df['psnr'], label='Validation PSNR', color='green', marker='o', linestyle='-')
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.set_title('Validation PSNR over Steps', fontsize=14)
        ax2.legend()
        ax2.grid(True, linestyle='--')

    # 4. 在第三个子图 (axes[2]) 上绘制 SSIM 曲线 
    if eval_df is not None and not eval_df.empty and 'ssim' in eval_df.columns:
        ax3 = axes[2]
        ax3.plot(eval_df['step'], eval_df['ssim'], label='Validation SSIM', color='red', marker='o', linestyle='-')
        ax3.set_xlabel('Training Step', fontsize=12)
        ax3.set_ylabel('SSIM', fontsize=12)
        ax3.set_title('Validation SSIM over Steps', fontsize=14)
        ax3.legend()
        ax3.grid(True, linestyle='--')

    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为大标题留出空间
    
    # 保存整个大图
    output_path = 'training_metrics_analysis.png'
    plt.savefig(output_path, dpi=300)
    print(f"分析图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()

if __name__ == '__main__':
    LOG_FILE_PATH = 'log.txt'
    
    # 解析日志文件，得到两个DataFrame
    train_dataframe, eval_dataframe = parse_log_file(LOG_FILE_PATH)
    
    # 将两个DataFrame传入绘图函数
    plot_metrics(train_dataframe, eval_dataframe)
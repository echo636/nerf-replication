import torch
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 检查命令行是否提供了文件路径
if len(sys.argv) < 2:
    print("Usage: python check_grid.py <path_to_occupancy_grid.pt>")
    sys.exit(1)

grid_path = sys.argv[1]

print(f"Loading grid from: {grid_path}")

try:
    # 加载占用网格
    occupancy_grid = torch.load(grid_path)

    # 1. 检查数据类型
    print(f"Data type: {occupancy_grid.dtype}")
    assert occupancy_grid.dtype == torch.bool, "Grid should be boolean!"

    # 2. 检查形状
    print(f"Shape: {occupancy_grid.shape}")
    assert len(occupancy_grid.shape) == 3, "Grid should be 3-dimensional!"

    # 3. 检查占用率 (再次确认)
    occupied_count = occupancy_grid.sum().item()
    total_voxels = occupancy_grid.numel()
    occupancy_rate = 100 * occupied_count / total_voxels
    print(f"Occupied voxels: {occupied_count}/{total_voxels} ({occupancy_rate:.2f}%)")

    print("\n✅ Grid inspection passed!")

    visualize = input("Do you want to visualize the grid as a 3D scatter plot? (y/n): ")
    if visualize.lower() == 'y':
        print("Preparing visualization... (This might take a moment for large grids)")
        
        # 找到所有被占用的体素的索引
        occupied_indices = torch.where(occupancy_grid)
        
        # 将索引转换为 numpy 数组以便绘图
        x = occupied_indices[0].cpu().numpy()
        y = occupied_indices[1].cpu().numpy()
        z = occupied_indices[2].cpu().numpy()
        
        if len(x) == 0:
            print("No occupied voxels to visualize.")
        else:
            # 为了性能，如果点太多，只绘制一部分
            max_points_to_plot = 50000
            if len(x) > max_points_to_plot:
                print(f"Too many points ({len(x)}). Plotting a random subset of {max_points_to_plot}.")
                import numpy as np
                subset_indices = np.random.choice(len(x), max_points_to_plot, replace=False)
                x = x[subset_indices]
                y = y[subset_indices]
                z = z[subset_indices]

            # 创建 3D 绘图
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(x, y, z, s=1) # s=1 设置点的大小
            
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
            ax.set_zlabel('Z index')
            ax.set_title('Occupancy Grid Visualization')
            
            # 让坐标轴比例一致
            max_range = occupancy_grid.shape[0]
            ax.set_box_aspect([1,1,1]) # 保证立方体看起来是立方体

            print("Displaying plot...")
            plt.show()

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
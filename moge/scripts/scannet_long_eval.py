import json
import matplotlib.pyplot as plt
import os

# 获取项目根目录（兼容Jupyter和普通脚本）
try:
    # 普通脚本环境
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
except NameError:
    # Jupyter环境，使用当前工作目录
    project_root = os.getcwd()
    # 如果当前目录不是项目根目录，尝试向上查找
    while not os.path.exists(os.path.join(project_root, "exps")):
        parent = os.path.dirname(project_root)
        if parent == project_root:  # 已经到达根目录
            break
        project_root = parent

# 定义要比较的方法列表（可以修改这里添加更多方法）
methods = [
    "flashdepth_long",
    # 可以添加更多方法，例如：
    # "conv_gru_final",
    # "mogev2",
    # "vda_long",
]

# 定义frame数量列表
frame_counts = [100, 200, 300, 400, 500]

# 定义不同的标记和颜色样式
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '+']
colors = plt.cm.tab10(range(len(methods)))

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 为每个方法读取数据并绘制
for method_idx, method_name in enumerate(methods):
    json_path = os.path.join(project_root, "exps/video_evaluation", method_name, "evaluation_results.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skipping {method_name}")
        continue
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取scannet_long的数据
    scannet_data = data.get("scannet_long", {})
    
    # 提取abs_rel和delta1_acc的mean值
    abs_rel_values = []
    delta1_acc_values = []
    
    for frame_count in frame_counts:
        # 提取abs_relative_difference
        abs_rel_key = f"abs_relative_difference_lstsq_{frame_count}frames"
        if abs_rel_key in scannet_data:
            abs_rel_values.append(scannet_data[abs_rel_key]["mean"])
        else:
            abs_rel_values.append(None)
        
        # 提取delta1_acc
        delta1_key = f"delta1_acc_lstsq_{frame_count}frames"
        if delta1_key in scannet_data:
            delta1_acc_values.append(scannet_data[delta1_key]["mean"])
        else:
            delta1_acc_values.append(None)
    
    # 绘制abs_rel
    ax1.plot(frame_counts, abs_rel_values, marker=markers[method_idx % len(markers)], 
             label=method_name, linewidth=2, markersize=8, color=colors[method_idx])
    
    # 绘制delta1_acc
    ax2.plot(frame_counts, delta1_acc_values, marker=markers[method_idx % len(markers)], 
             label=method_name, linewidth=2, markersize=8, color=colors[method_idx])

# 设置abs_rel子图
ax1.set_xlabel('Frame Count', fontsize=12)
ax1.set_ylabel('abs_rel', fontsize=12)
ax1.set_title('abs_rel vs Frame Count', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 设置delta1_acc子图
ax2.set_xlabel('Frame Count', fontsize=12)
ax2.set_ylabel('delta1_acc', fontsize=12)
ax2.set_title('delta1_acc vs Frame Count', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 定义颜色映射和标签字典
color_map = ['tab:purple', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue']
label_dict = {
    0: 'Real',
    1: 'Deepfakes',
    2: 'Face2Face',
    3: 'FaceSwap',
    4: 'NeuralTextures',
}

# 定义t-SNE函数
def tsne_draw(x_transformed, numerical_labels, ax, epoch=0, log='', detector_name=None):
    labels = [label_dict[label] for label in numerical_labels]

    tsne_df = pd.DataFrame(x_transformed, columns=['X', 'Y'])
    tsne_df["Targets"] = labels
    tsne_df["NumericTargets"] = numerical_labels
    tsne_df.sort_values(by="NumericTargets", inplace=True)
    marker_list = ['*' if label == 0 else 'o' for label in tsne_df["NumericTargets"]]

    for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], [color_map[i] for i in tsne_df["NumericTargets"]], marker_list):
        ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m)

    print(f'epoch{epoch} ' + log)
    ax.axis('off')

# 用户定义的每种伪造方式的样本数量
num_samples = 1500  # 你可以根据需要调整这个值
detector = 'PDP'

# 文件路径列表
detector_name_list = [
    '../training/logs/my_detection/mydetector_2024-10-03-12-53-59/test/FF-DF/feat_best.npy',
    '../training/logs/my_detection/mydetector_2024-10-03-12-53-59/test/FF-F2F/feat_best.npy',
    '../training/logs/my_detection/mydetector_2024-10-03-12-53-59/test/FF-FS/feat_best.npy',
    '../training/logs/my_detection/mydetector_2024-10-03-12-53-59/test/FF-NT/feat_best.npy',
    '../training/logs/my_detection/mydetector_2024-10-03-12-53-59/test/FaceForensics++/feat_best.npy',
]

# 初始化t-SNE
tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=12, random_state=1024, learning_rate=250)

# 创建绘图区域
fig, axs = plt.subplots(1, 1, figsize=(10, 8))  # 只绘制一个子图

all_features = []
all_labels = []

# 处理每个伪造方式文件，抽取相同数量的伪造人脸样本
for i, file_path in enumerate(detector_name_list[:-1]):  # 不包括最后一个文件（真实人脸）
    print(f'Processing {file_path}...')
    
    # 加载数据
    tsne_dict = np.load(file_path, allow_pickle=True).item()

    # 获取特征数据
    features = tsne_dict['feature_art']
    labels = tsne_dict['label']

    # 确保特征是NumPy数组
    if isinstance(features, list):
        features = np.array(features)
    if isinstance(labels, list):
        labels = np.array(labels)

    print(features.shape)
    # 随机选择样本
    fake_mask = labels == 1
    available_samples = np.sum(fake_mask)
    fake_samples = min(num_samples, available_samples)  # 使用最小值以避免超出范围
    print("fake:", fake_samples)
    idx = np.random.choice(np.where(fake_mask)[0], size=fake_samples, replace=False)
    sampled_features = features[idx]
    sampled_labels = labels[idx]

    # 将伪造类型映射到对应的标签值
    if 'FF-DF' in file_path:
        sampled_labels[:] = 1  # 设置标签为 1
    elif 'FF-F2F' in file_path:
        sampled_labels[:] = 2  # 设置标签为 2
    elif 'FF-FS' in file_path:
        sampled_labels[:] = 3  # 设置标签为 3
    elif 'FF-NT' in file_path:
        sampled_labels[:] = 4  # 设置标签为 4

    all_features.append(sampled_features)
    all_labels.append(sampled_labels)

# 处理真实人脸样本
real_file_path = detector_name_list[-1]
print(f'Processing {real_file_path}...')

# 加载数据
tsne_dict = np.load(real_file_path, allow_pickle=True).item()
features = tsne_dict['feature_art']
labels = tsne_dict['label']

# 确保特征是NumPy数组
if isinstance(features, list):
    features = np.array(features)
if isinstance(labels, list):
    labels = np.array(labels)

# 随机选择真实人脸样本
real_mask = labels == 0
available_real_samples = np.sum(real_mask)
total_real_samples = fake_samples * 2
real_samples = min(total_real_samples, available_real_samples)  # 使用最小值以避免超出范围
print("real:", real_samples)
idx = np.random.choice(np.where(real_mask)[0], size=real_samples, replace=False)
real_features = features[idx]
real_labels = labels[idx]

all_features.append(real_features)
all_labels.append(real_labels)

# 整合所有特征和标签
all_features = np.vstack(all_features)
all_labels = np.hstack(all_labels)

# 对数据进行t-SNE降维
feat_transformed = tsne.fit_transform(all_features)

# 绘制散点图
tsne_draw(feat_transformed, all_labels, ax=axs, epoch='n', log='features visualize', detector_name='PDP')

# 创建图例
handles = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color_map[i], markersize=10) if i == 0 else 
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=10) for i in range(5)]
labels = [label_dict[i] for i in range(5)]
fig.legend(handles, labels, loc="upper right", fontsize=12)

plt.tight_layout()
output_dir = '/mnt/raid1/zwj22/paper_models/DeepfakeBench-v2-main/training/outputs/tsne_imgs/'
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
plt.savefig(os.path.join(output_dir, f"5cls_{detector}_{total_real_samples}.png"))
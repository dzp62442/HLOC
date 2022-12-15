# %%

import os
from pathlib import Path
import pycolmap
from tqdm import tqdm

from hloc.utils import viz_3d, parsers
from hloc.visualization import plot_images, read_image
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, pairs_from_exhaustive, localize_sfm

# %%
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## 对 SportsHall_Total 数据集进三维重建（包括参考图像和查询图像）作为真值

# %% [markdown]
# 配置路径和参数

# %%
my_datasets = ['House11', 'SportsHall', 'SportsHall_Total']  # 11号楼、体育馆
sfm_dataset = Path('datasets/' + my_datasets[2])  # 用于三维重建的数据集

img_names = [r.relative_to(sfm_dataset).as_posix() for r in sfm_dataset.iterdir()]  # 相对于数据集根目录的文件名

outputs = sfm_dataset / 'hloc'  # 与官方不同，将outputs放置在数据集文件夹中

sfm_dir = outputs / 'Total_sfm_spsg'  # 对参考图像和查询图像一起重建得到的 SfM 模型
sfm_pairs = outputs / 'sfm_pairs.txt'
loc_pairs = outputs / 'loc_pairs.txt'

retrieval = outputs / 'retrieval.h5'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

intrinsics = outputs / 'imgs_with_intrinsics.txt'  # 保存所有图像名称和估计的相机参数
results = outputs / 'SportsHall_Total_hloc_superpoint+superglue_netvlad20.txt'  # the result file

# %% [markdown]
# 通过图像检索在全部图像中寻找图像对

# %%
# 由于数据集相对较大，使用NetVLAD提取全局描述符并寻找与每幅图像最相似的图像
# 若数据集较小，直接进行穷举匹配
extract_features.main(retrieval_conf, sfm_dataset, feature_path=retrieval)
pairs_from_retrieval.main(retrieval, sfm_pairs, num_matched=10)  # 配对图像数目应为10-20，追求速度可用5

# %% [markdown]
# 提取并匹配全部图像的局部特征

# %%
extract_features.main(feature_conf, sfm_dataset, feature_path=features)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# %% [markdown]
# 对全部图像运行COLMAP

# %%
total_model = reconstruction.main(sfm_dir, sfm_dataset, sfm_pairs, features, matches)

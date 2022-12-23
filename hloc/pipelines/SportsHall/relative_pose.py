from typing import Dict, List, Union, Optional, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pycolmap

from hloc import logger
from hloc.utils.io import get_keypoints, get_matches
from hloc.utils.parsers import parse_image_dict, parse_retrieval


#! 求解 img1 到 img2 的变换矩阵 T_1_2（四元数和平移向量表示）
def main(dataset_root_dir: Path,  # 数据集根目录
         intrinsics: Union[Path, list],  # 相机内参文件路径
         pairs: Path,  # 待估计的图像配对
         features: Path,
         matches: Path,
         results: Path,
         options: Optional[Dict[str, Any]] = {},  # 从图像中推断相机内参时指定相机模型
         config: Dict = None):
    
    assert pairs.exists(), pairs
    assert features.exists(), features
    assert matches.exists(), matches   

    logs = {
        'intrinsics': intrinsics,
        'pairs': pairs,
        'features': features,
        'matches': matches,
        'relative': {},
        'success': 0,
        'failed': 0,
    }

    if not isinstance(intrinsics, list):
        intrinsics = [intrinsics]
    cameras = parse_image_dict(intrinsics, with_intrinsics=True)

    pairs_dict = parse_retrieval(pairs)  # 以字典格式存储的待估计的图像的配对
    logger.info('Start two view geometry estimation...')

    for img1 in tqdm(pairs_dict.keys()):
        logs['relative'][img1] = {}
        for i, img2 in enumerate(pairs_dict[img1]):         
            # 加载匹配特征点
            m, scores = get_matches(matches, img1, img2)
            kp1 = get_keypoints(features, img1)
            kp2 = get_keypoints(features, img2)
            kp1 = kp1[m[:,0]]
            kp2 = kp2[m[:,1]]
            # 加载相机内参
            cam1 = cameras[img1]
            cam2 = cameras[img2]
            # 估计相对位姿
            ret = pycolmap.two_view_geometry_estimation(kp1, kp2, cam1, cam2)
            if(ret['success']):
                logs['relative'][img1][img2] = ret
                logs['success'] += 1
            else:
                logger.warning(f'{img1} and {img2} estimate failed. Skipping...')
                logs['failed'] += 1
                continue
            #print(i, img1, img2)            

    # 保存估计结果
    logger.info(f'Estimation success / failed: {logs["success"]} / {logs["failed"]}')
    logger.info(f'Writing relative poses estimation to {results}...')
    with open(results, 'w') as f:
        for img1 in logs['relative'].keys():
            for img2 in logs['relative'][img1].keys():
                qvec, tvec = logs['relative'][img1][img2]['qvec'], logs['relative'][img1][img2]['tvec']
                qvec = ' '.join(map(str, qvec))
                tvec = ' '.join(map(str, tvec))
                f.write(f'{img1} {img2} {qvec} {tvec}\n')

    return logs
    
    

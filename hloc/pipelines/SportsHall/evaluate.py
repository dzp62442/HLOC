from typing import Dict, List, Union, Optional, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pycolmap
import cv2

from hloc import logger

def read_poses_text(paths: Union[Path, list], isAbsolute: bool):
    if not isinstance(paths, list):
        paths = [paths]
    poses = {}
    for path in paths:
        assert path.exists(), path
        with open(path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                if len(line) == 0 or line[0] == '#':
                    continue
                if(isAbsolute):  # 读取绝对位姿文件
                    name, *data = line.split()
                    data = np.array(data, dtype=np.float64)
                    poses[name] = {}
                    poses[name]['qvec'] = data[0:4]
                    poses[name]['tvec'] = data[4:7]
                else:  # 读取相对位姿文件
                    name1, name2, *data = line.split()
                    data = np.array(data, dtype=np.float64)
                    if (name1 not in poses.keys()):
                        poses[name1] = {}
                    poses[name1][name2] = {}
                    poses[name1][name2]['qvec'] = data[0:4]
                    poses[name1][name2]['tvec'] = data[4:7]
    return poses

def compute_error(R, t, R_gt, t_gt, nomarlize=True):
    if (nomarlize):
        t = t / np.linalg.norm(t)
        t_gt = t_gt / np.linalg.norm(t_gt)
    
    #! 以下为 HLoc 的 Cambridge 和 7Scenes 数据集采用的误差定义
    # e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    # cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    # e_R = np.rad2deg(np.abs(np.arccos(cos)))

    #! 以下为 DSAC* 中的误差定义
    e_t = np.linalg.norm(t - t_gt)
    e_R = np.matmul(R, np.transpose(R_gt))
    e_R = cv2.Rodrigues(e_R)[0]
    e_R = np.linalg.norm(e_R) * 180 / np.pi

    '''
    经测试以上两种误差计算方法 角度误差完全一致
    平移误差有微小差别 平移误差定位率的差别约为0.5%
    第二种平移误差计算方法更常见，定位率略高
    '''
    
    return e_R, e_t

def evaluate_absolute_poses(prediction: Union[Path, list], ground_truth: Union[Path, list]): 
    abs_poses_pred = read_poses_text(prediction, isAbsolute=False)  # 预测结果为绝对位姿
    abs_poses_gt = read_poses_text(ground_truth, isAbsolute=True)  # 真值为绝对位姿

    errors_R, errors_t = [], []
    logger.info('Start evaluate absolute poses estimation...')

    for img in tqdm(abs_poses_pred.keys()):
        R = pycolmap.qvec_to_rotmat(abs_poses_pred[img]['qvec'])
        t = abs_poses_pred[img]['tvec']
        R_gt = pycolmap.qvec_to_rotmat(abs_poses_gt[img]['qvec'])
        t_gt = abs_poses_gt[img]['tvec']

        e_R, e_t = compute_error(R, t, R_gt, t_gt, nomarlize=True)

        errors_R.append(e_R)
        errors_t.append(e_t)
    
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    return errors_R, errors_t

def evaluate_relative_poses(prediction: Union[Path, list], ground_truth: Union[Path, list]): 
    rel_poses_pred = read_poses_text(prediction, isAbsolute=False)  # 预测结果为相对位姿
    abs_poses_gt = read_poses_text(ground_truth, isAbsolute=True)  # 真值为绝对位姿

    errors_R, errors_t = [], []
    logger.info('Start evaluate relative poses estimation...')

    for img1 in tqdm(rel_poses_pred.keys()):
        for img2 in rel_poses_pred[img1].keys():
            R = pycolmap.qvec_to_rotmat(rel_poses_pred[img1][img2]['qvec'])
            t = rel_poses_pred[img1][img2]['tvec']
            qvec_gt, t_gt = pycolmap.relative_pose(abs_poses_gt[img1]['qvec'], abs_poses_gt[img1]['tvec'], 
                                                   abs_poses_gt[img2]['qvec'], abs_poses_gt[img2]['tvec'])
            R_gt = pycolmap.qvec_to_rotmat(qvec_gt)

            e_R, e_t = compute_error(R, t, R_gt, t_gt, nomarlize=True)

            errors_R.append(e_R)
            errors_t.append(e_t)
    
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    return errors_R, errors_t
    
def main(prediction: Union[Path, list], ground_truth: Union[Path, list], isAbsolute: bool):
    if (isAbsolute):  # 估计结果为绝对位姿
        errors_R, errors_t = evaluate_absolute_poses(prediction, ground_truth)
    else:  # 估计结果为相对位姿
        errors_R, errors_t = evaluate_relative_poses(prediction, ground_truth)
    
    mean_R, mean_t = np.mean(errors_R), np.mean(errors_t)
    med_R, med_t = np.median(errors_R), np.median(errors_t)
    max_R, max_t = np.max(errors_R), np.max(errors_t)

    logger.info(f'Mean errors: {mean_R:.3f}deg, {mean_t:.3f}unit')
    logger.info(f'Median errors: {med_R:.3f}deg, {med_t:.3f}unit')
    logger.info(f'Max errors: {max_R:.3f}deg, {max_t:.3f}unit')

    threshs_R = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5]

    for th_R in threshs_R:
        ratio = np.mean(errors_R < th_R)
        logger.info(f'{th_R:.1f}deg : {ratio*100:.2f}%') 
    
    for th_t in threshs_t:
        ratio = np.mean(errors_t < th_t)
        logger.info(f'{th_t:.3f}unit : {ratio*100:.2f}%') 
    

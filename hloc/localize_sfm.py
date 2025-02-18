import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union, Optional, Any
from tqdm import tqdm
import pickle
import pycolmap

from . import logger
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_image_lists, parse_retrieval


def do_covisibility_clustering(frame_ids: List[int],
                               reconstruction: pycolmap.Reconstruction):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed if p2D.has_point3D()
                for obs in
                reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        ret = pycolmap.absolute_pose_estimation(
            points2D, points3D, query_camera,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        return ret


def pose_from_cluster(
        localizer: QueryLocalizer,
        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        features_path: Path,
        matches_path: Path,
        **kwargs):

    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D() == 0:
            logger.debug(f'No 3D points found for {image.name}.')
            continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                 for p in image.points2D])

        matches, _ = get_matches(matches_path, qname, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    ret['camera'] = {
        'model': query_camera.model_name,
        'width': query_camera.width,
        'height': query_camera.height,
        'params': query_camera.params,
    }

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]
    log = {
        'db': db_ids,
        'PnP_ret': ret,  # log['PnP_ret'] 就是 ret
        'keypoints_query': kpq[mkp_idxs],
        'points3D_ids': mp3d_ids,
        'points3D_xyz': None,  # we don't log xyz anymore because of file size
        'num_matches': num_matches,
        'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def main(reference_sfm: Union[Path, pycolmap.Reconstruction],  # pycolmap.Reconstruction
         queries: Path,  # queries_with_intrinsics.txt
         loc_pairs: Path,  # loc_pairs
         features: Path,
         matches: Path,
         results: Path,
         ransac_thresh: int = 12,
         covisibility_clustering: bool = False,
         prepend_camera_name: bool = False,
         config: Dict = None):

    assert loc_pairs.exists(), loc_pairs
    assert features.exists(), features
    assert matches.exists(), matches

    #! 传入相机内参
    queries = parse_image_lists(queries, with_intrinsics=True)
    loc_pairs_dict = parse_retrieval(loc_pairs)

    logger.info('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    ref_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    localizer = QueryLocalizer(reference_sfm, config)

    cameras, poses = {}, {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': loc_pairs,
        'loc': {},
    }
    logger.info('Starting localization...')
    for qname, qcam in tqdm(queries):
        if qname not in loc_pairs_dict:
            logger.warning(
                f'No images retrieved for query image {qname}. Skipping...')
            continue
        ref_names = loc_pairs_dict[qname]
        ref_ids = []
        for name in ref_names:
            if name not in ref_name_to_id:
                logger.warning(f'Image {name} was retrieved but not in database')
                continue
            ref_ids.append(ref_name_to_id[name])

        if covisibility_clustering:  #!目前没用过
            clusters = do_covisibility_clustering(ref_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                        localizer, qname, qcam, cluster_ids, features, matches)
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'ref': ref_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering,
            }
        else:  #!目前常用
            # log['PnP_ret'] 与 ret 相同
            ret, log = pose_from_cluster(localizer, qname, qcam, ref_ids, features, matches)
            cameras[qname] = qcam
            if ret['success']:
                poses[qname] = pycolmap.Image(qvec=ret['qvec'], tvec=ret['tvec'])
            else:  # 位姿估计失败，将最相似的配对图像作为查询图像的位姿
                closest = reference_sfm.images[ref_ids[0]]
                poses[qname] = pycolmap.Image(qvec=closest.tvec, tvec=closest.qvec)
            log['covisibility_clustering'] = covisibility_clustering
            log['inlier_rate'] = f'{ret["num_inliers"]}/{len(ret["inliers"])}'
            logs['loc'][qname] = log

    logger.info(f'Localized {len(poses)} / {len(queries)} images.')
    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for qname, qcam in queries:
            qvec, tvec = logs['loc'][qname]['PnP_ret']['qvec'], logs['loc'][qname]['PnP_ret']['tvec']
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            f.write(f'{qname} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')  # 保存logs
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')

    return poses, cameras, logs


def main2(dataset_root_dir: Path,  # 数据集根目录
          reference_sfm: Union[Path, pycolmap.Reconstruction],  # pycolmap.Reconstruction实例或路径
          query_names: list,
          loc_pairs: Path,
          features: Path,
          matches: Path,
          results: Path,
          covisibility_clustering: bool = False,
          prepend_camera_name: bool = False,  #?这是啥，有啥用
          options: Optional[Dict[str, Any]] = {},  # 从图像中推断相机内参时指定相机模型
          config: Dict = None):

    assert loc_pairs.exists(), loc_pairs
    assert features.exists(), features
    assert matches.exists(), matches

    loc_pairs_dict = parse_retrieval(loc_pairs)  # 以字典格式存储的查询图像与参考图像的配对

    logger.info('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):  # 不是colmap模型，而是sfm_dir路径
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    ref_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}  # 所有参考图像对应的id

    localizer = QueryLocalizer(reference_sfm, config)

    cameras, poses = {}, {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': loc_pairs,
        'loc': {},
    }
    logger.info('Starting localization...')
    for i, qname in tqdm(enumerate(query_names)):
        if qname not in loc_pairs_dict:  # 查询图像未配对，跳过该图像
            logger.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue
        #! 从图像中推断相机内参
        qcam = pycolmap.infer_camera_from_image(dataset_root_dir / qname, options)
        ref_names = loc_pairs_dict[qname]
        ref_ids = []  # 与查询图像配对的参考图像的ids
        for name in ref_names:
            if name not in ref_name_to_id:
                logger.warning(f'Image {name} was retrieved but not in reference database')
                continue
            ref_ids.append(ref_name_to_id[name])

        if covisibility_clustering:  #!目前没用过
            clusters = do_covisibility_clustering(ref_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(localizer, qname, qcam, cluster_ids, features, matches)
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'ref': ref_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering  # True
            }
        else:  #!目前常用
            # log['PnP_ret'] 与 ret 相同
            ret, log = pose_from_cluster(localizer, qname, qcam, ref_ids, features, matches)
            cameras[qname] = qcam
            if ret['success']:
                poses[qname] = pycolmap.Image(qvec=ret['qvec'], tvec=ret['tvec'])
            else:  # 位姿估计失败，将最相似的配对图像作为查询图像的位姿
                closest = reference_sfm.images[ref_ids[0]]
                poses[qname] = pycolmap.Image(qvec=closest.tvec, tvec=closest.qvec)
            log['covisibility_clustering'] = covisibility_clustering  # False
            log['inlier_rate'] = f'{ret["num_inliers"]}/{len(ret["inliers"])}'
            logs['loc'][qname] = log

    logger.info(f'Localized {len(poses)} / {len(query_names)} images.')
    logger.info(f'Writing poses to {results}...')  # 保存位姿估计结果
    with open(results, 'w') as f:
        for qname in query_names:
            qvec, tvec = logs['loc'][qname]['PnP_ret']['qvec'], logs['loc'][qname]['PnP_ret']['tvec']
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            f.write(f'{qname} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')  # 保存logs
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')
    
    return poses, cameras, logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)

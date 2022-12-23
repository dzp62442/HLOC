from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
import pycolmap
from typing import Dict, List

logger = logging.getLogger(__name__)


def parse_image_list(path, with_intrinsics=False):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if with_intrinsics:
                model, width, height, *params = data
                params = np.array(params, float)
                cam = pycolmap.Camera(model, int(width), int(height), params)
                images.append((name, cam))
            else:
                images.append(name)

    assert len(images) > 0
    logger.info(f'Imported {len(images)} images from {path.name}')
    return images


def parse_image_lists(paths, with_intrinsics=False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    print(files)
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics)
    return images


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')

def parse_image_dict(paths: list, with_intrinsics=True):
    images = {}  # 以字典格式保存图像名称和内参
    for path in paths:
        assert path.exists(), path
        with open(path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                if len(line) == 0 or line[0] == '#':
                    continue
                name, *data = line.split()
                if with_intrinsics:
                    model, width, height, *params = data
                    params = np.array(params, float)
                    images[name] = pycolmap.Camera(model, int(width), int(height), params)
                else:
                    images[name] = 0

        assert len(images) > 0
        logger.info(f'Imported {len(images)} images from {path.name}')
    return images
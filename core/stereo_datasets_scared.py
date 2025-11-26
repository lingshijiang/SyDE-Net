import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
from PIL import Image

import sys

sys.path.append(os.getcwd())

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
try:
    # 可选的数据预处理与列表读取，若不存在则忽略
    from .data_io import get_transform, read_all_lines
except Exception:
    get_transform = None
    read_all_lines = None


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        # print(f"disp shape: {disp.shape}")

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:

                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]
        # print(f"flow shape: {flow.shape}")
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/e/MonSter-main/core/data2/cjd/StereoDatasets/sceneflow',
                 dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        assert os.path.exists(root)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        # left_images = sorted( glob(osp.join(root, lf.dstype, split, '*/*/left/*.png')) )
        # right_images = [ im.replace('left', 'right') for im in left_images ]
        right_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/right/*.png')))
        left_images = [im.replace('right', 'left') for im in right_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        # left_images = sorted( glob(osp.join(root, self.dstype, split, '*/left/*.png')) )
        # right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        right_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/right/*.png')))
        left_images = [im.replace('right', 'left') for im in right_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        # left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/*/left/*.png')) )
        # right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        right_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/right/*.png')))
        left_images = [im.replace('right', 'left') for im in right_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/eth3d', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) if split == 'training' else [
                                                                                                                       osp.join(
                                                                                                                           root,
                                                                                                                           'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(
            image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted(glob(osp.join(root, 'training/*_left/*/frame_*.png')))
        image2_list = sorted(glob(osp.join(root, 'training/*_right/*/frame_*.png')))
        disp_list = sorted(glob(osp.join(root, 'training/disparities/*/frame_*.png'))) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/data_wxq/fallingthings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        image1_list += sorted(glob(root + '/*/*/*/*left.jpg'))
        image2_list += sorted(glob(root + '/*/*/*/*right.jpg'))
        disp_list += sorted(glob(root + '/*/*/*/*left.depth.png'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e
                     in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/kitti/2015', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        root_12 = '/data2/cjd/StereoDatasets/kitti/2012/'
        image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
        disp_list = sorted(
            glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(
            root, 'training/disp_occ/000085_10.png')] * len(image1_list)

        root_15 = '/data2/cjd/StereoDatasets/kitti/2015/'
        image1_list += sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
        image2_list += sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
        disp_list += sorted(
            glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(
            root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='/data/cjd/stereo_dataset/vkitti2/'):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispVKITTI2)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/middlebury', split='2014', resolution='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [str(scene / "disp0.pfm")]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [
                            [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]


class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/data_wxq/instereo2k'):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/train/*/*/left.png') + glob(root + '/test/*/left.png'))
        image2_list = sorted(glob(root + '/train/*/*/right.png') + glob(root + '/test/*/right.png'))
        disp_list = sorted(glob(root + '/train/*/*/left_disp.png') + glob(root + '/test/*/left_disp.png'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class CARLA(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/carla-highres'):
        super(CARLA, self).__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/trainingF/*/im0.png'))
        image2_list = sorted(glob(root + '/trainingF/*/im1.png'))
        disp_list = sorted(glob(root + '/trainingF/*/disp0GT.pfm'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class DrivingStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/drivingstereo/', image_set='rainy'):
        reader = frame_utils.readDispDrivingStereo_half
        super().__init__(aug_params, sparse=True, reader=reader)
        assert os.path.exists(root)
        image1_list = sorted(glob(os.path.join(root, image_set, 'left-image-half-size/*.jpg')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'right-image-half-size/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, image_set, 'disparity-map-half-size/*.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class SCARED(StereoDataset):
    def __init__(self, aug_params=None, root=None, split='train', train_ratio=0.8):
        """
        SCARED数据集类
        Args:
            aug_params: 数据增强参数
            root: 数据集根目录
            split: 'train' 或 'val'，用于训练集或验证集
            train_ratio: 训练集比例，默认0.8（80%训练，20%验证）
        """
        # 自动检测路径
        if root is None:
            # 尝试不同的路径，优先使用Windows路径
            possible_roots = [
                'E:/scared_toolkit-master',  # Windows路径
                'E:\\scared_toolkit-master',  # Windows路径（反斜杠）
                '/mnt/e/scared_toolkit-master',  # WSL路径
            ]
            
            root = None
            for possible_root in possible_roots:
                print(f"[SCARED] 尝试路径: {possible_root}")
                print(f"[SCARED] 路径存在性: {os.path.exists(possible_root)}")
                
                if os.path.exists(possible_root):
                    # 进一步检查关键文件是否存在
                    test_files_check = os.path.join(possible_root, 'test_files.txt')
                    processed_data_check = os.path.join(possible_root, 'processed_data')
                    print(f"[SCARED] 检查关键文件:")
                    print(f"  - test_files.txt存在: {os.path.exists(test_files_check)}")
                    print(f"  - processed_data存在: {os.path.exists(processed_data_check)}")
                    
                    if os.path.exists(test_files_check) and os.path.exists(processed_data_check):
                        # 测试能否访问深层子目录
                        test_subdir = os.path.join(processed_data_check, 'dataset_3', 'keyframe_4', 'data', 'left_rectified')
                        print(f"[SCARED] 测试深层目录访问: {os.path.exists(test_subdir)}")
                        
                        if os.path.exists(test_subdir):
                            # 测试具体文件访问
                            test_file = os.path.join(test_subdir, '000390.png')
                            print(f"[SCARED] 测试文件访问: {os.path.exists(test_file)}")
                            
                            if os.path.exists(test_file):
                                root = possible_root
                                if possible_root.startswith('/mnt'):
                                    print(f"[SCARED] 选择WSL路径: {root}")
                                else:
                                    print(f"[SCARED] 选择Windows路径: {root}")
                                break
                    else:
                        print(f"[SCARED] 跳过路径 {possible_root}，关键文件不存在")
            
            if root is None:
                raise ValueError(f"SCARED数据集未找到，尝试了以下路径: {possible_roots}")
        
        print(f"[SCARED] 开始初始化，根目录: {root}, 分割: {split}")
        super(SCARED, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispSCARED)
        print(f"[SCARED] 父类初始化完成")
        
        print(f"[SCARED] 检查根目录是否存在: {root}")
        assert os.path.exists(root), f"SCARED dataset root {root} does not exist"
        print(f"[SCARED] 根目录存在")
        
        # 存储分割信息
        self.split = split
        self.train_ratio = train_ratio
        
        # 读取test_files.txt文件
        test_files_path = os.path.join(root, 'test_files.txt')
        print(f"[SCARED] 检查test_files.txt路径: {test_files_path}")
        assert os.path.exists(test_files_path), f"test_files.txt not found at {test_files_path}"
        print(f"[SCARED] test_files.txt存在")
        
        # 检查是否有拆分后的深度文件目录
        self.gt_depths_split_dir = os.path.join(root, 'gt_depths_split')
        if os.path.exists(self.gt_depths_split_dir):
            print(f"[SCARED] 找到拆分后的深度文件目录: {self.gt_depths_split_dir}")
            self.use_split_files = True
        else:
            # 如果没有拆分文件，回退到原始npz文件
            gt_depths_path = os.path.join(root, 'gt_depths.npz')
            print(f"[SCARED] 未找到拆分文件，使用原始gt_depths.npz: {gt_depths_path}")
            assert os.path.exists(gt_depths_path), f"gt_depths.npz not found at {gt_depths_path}"
            self.gt_depths_path = gt_depths_path
            self.use_split_files = False
            print(f"[SCARED] 将在首次访问时加载gt_depths.npz")
        
        processed_data_root = os.path.join(root, 'processed_data')
        print(f"[SCARED] processed_data路径: {processed_data_root}")
        
        # 解析test_files.txt
        print(f"[SCARED] 正在读取test_files.txt...")
        with open(test_files_path, 'r') as f:
            lines = f.readlines()
        
        print(f"[SCARED] test_files.txt包含 {len(lines)} 行")
        
        # 数据分割：80%训练，20%验证
        total_samples = len(lines)
        train_count = int(total_samples * self.train_ratio)
        
        # 使用固定种子确保分割的一致性
        import random
        random.seed(42)  # 固定种子
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        if self.split == 'train':
            selected_indices = indices[:train_count]
            print(f"[SCARED] 训练集: 使用前 {train_count} 个样本 (总共{total_samples}个)")
        else:  # val
            selected_indices = indices[train_count:]
            print(f"[SCARED] 验证集: 使用后 {total_samples - train_count} 个样本 (总共{total_samples}个)")
        
        processed_count = 0
        skipped_count = 0
        
        for enum_idx, idx in enumerate(selected_indices):
            line = lines[idx]
            if enum_idx < 10:  # 只打印前10行的详细信息
                print(f"[SCARED] 处理第 {enum_idx} 行 (原始索引{idx}): {line.strip()}")
            elif enum_idx == 10:
                print(f"[SCARED] ... (省略剩余行的详细输出)")
                
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    # 格式: dataset3/keyframe4	390	l
                    dataset_keyframe = parts[0]  # dataset3/keyframe4
                    frame_idx = int(parts[1])    # 390
                    
                    # 解析dataset和keyframe
                    dataset_name, keyframe_name = dataset_keyframe.split('/')
                    
                    # 处理dataset名称的不一致问题（test_files.txt中是dataset3，但实际目录是dataset_3）
                    if dataset_name.startswith('dataset') and '_' not in dataset_name:
                        # 如果是datasetX格式，转换为dataset_X格式
                        dataset_num = dataset_name.replace('dataset', '')
                        actual_dataset_name = f"dataset_{dataset_num}"
                    else:
                        actual_dataset_name = dataset_name
                    
                    # 处理keyframe名称的不一致问题（如果需要）
                    if keyframe_name.startswith('keyframe') and '_' not in keyframe_name:
                        # 如果是keyframeX格式，转换为keyframe_X格式
                        keyframe_num = keyframe_name.replace('keyframe', '')
                        actual_keyframe_name = f"keyframe_{keyframe_num}"
                    else:
                        actual_keyframe_name = keyframe_name
                    
                    # 构建图像路径
                    frame_id_str = f"{frame_idx:06d}"  # 000390
                    left_img_path = os.path.join(processed_data_root, actual_dataset_name, actual_keyframe_name, 
                                                'data', 'left_rectified', f"{frame_id_str}.png")
                    right_img_path = os.path.join(processed_data_root, actual_dataset_name, actual_keyframe_name, 
                                                 'data', 'right_rectified', f"{frame_id_str}.png")
                    
                    if enum_idx < 5:  # 只打印前5个路径的详细信息
                        print(f"[SCARED] 路径映射:")
                        print(f"  原始: {dataset_name}/{keyframe_name}")
                        print(f"  实际: {actual_dataset_name}/{actual_keyframe_name}")
                        print(f"  左图路径: {left_img_path}")
                        print(f"  右图路径: {right_img_path}")
                    
                    # 检查文件是否存在
                    if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                        self.image_list += [[left_img_path, right_img_path]]
                        
                        # 根据是否使用拆分文件来决定disparity路径
                        if self.use_split_files:
                            # 使用拆分后的单独文件
                            depth_filename = f"{dataset_keyframe.replace('/', '_')}_{frame_idx:06d}.npy"
                            depth_file_path = os.path.join(self.gt_depths_split_dir, depth_filename)
                            if os.path.exists(depth_file_path):
                                self.disparity_list += [depth_file_path]
                                processed_count += 1
                            else:
                                if enum_idx < 5:
                                    print(f"[SCARED] 跳过第 {enum_idx} 行，深度文件不存在: {depth_file_path}")
                                skipped_count += 1
                                # 移除刚添加的image_list项
                                self.image_list.pop()
                        else:
                            # 使用原始npz文件的索引 - 注意这里使用原始索引
                            self.disparity_list += [idx]
                            processed_count += 1
                    else:
                        if enum_idx < 5:  # 只打印前5个缺失文件的信息
                            print(f"[SCARED] 跳过第 {enum_idx} 行，文件不存在:")
                            print(f"  左图: {left_img_path} ({'存在' if os.path.exists(left_img_path) else '不存在'})")
                            print(f"  右图: {right_img_path} ({'存在' if os.path.exists(right_img_path) else '不存在'})")
                        skipped_count += 1
        
        # 如果使用原始npz文件，初始化延迟加载
        if not self.use_split_files:
            self._gt_depths_cache = None
            self._gt_depths_loaded = False
        
        print(f"[SCARED] 处理完成:")
        print(f"  - 成功处理: {processed_count} 个样本")
        print(f"  - 跳过: {skipped_count} 个样本")
        print(f"  - 总计: {len(self.image_list)} 个有效图像对")
        
        logging.info(f"Loaded {len(self.image_list)} image pairs from SCARED dataset")
    
    def _load_gt_depths_if_needed(self):
        """按需加载gt_depths数据（仅用于原始npz文件）"""
        if not self._gt_depths_loaded:
            print(f"[SCARED] 首次访问，正在加载gt_depths.npz...")
            gt_data = np.load(self.gt_depths_path)
            self._gt_depths_cache = gt_data['data']
            self._gt_depths_loaded = True
            print(f"[SCARED] gt_depths加载完成，形状: {self._gt_depths_cache.shape}")
        return self._gt_depths_cache
    
    def _load_single_depth(self, depth_path):
        """加载单个深度文件"""
        return np.load(depth_path)
    
    def __getitem__(self, index):
        # 重写__getitem__方法来处理gt_depths
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index] if index < len(self.extra_info) else ""

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        
        # 根据使用的深度文件类型加载深度数据
        if self.use_split_files:
            # 从单独的.npy文件加载
            depth_path = self.disparity_list[index]
            disp = self._load_single_depth(depth_path).astype(np.float32)
        else:
            # 从原始npz文件加载
            gt_depths = self._load_gt_depths_if_needed()
            original_idx = self.disparity_list[index]
            disp = gt_depths[original_idx].astype(np.float32)
        
        valid = disp > 0  # 大于0的像素认为是有效的

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # 将深度转换为视差（这里可能需要根据SCARED数据集的标定参数调整）
        # 暂时直接使用深度值作为视差
        disp = np.array(disp).astype(np.float32)

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # 处理灰度图像
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]
        disparity_info = self.disparity_list[index] if self.use_split_files else f"gt_depths_{self.disparity_list[index]}"
        return self.image_list[index] + [disparity_info], img1, img2, flow, valid.float()


class SCAREDListDataset(StereoDataset):
    """
    使用基于列表文件(list file)的SCARED数据集读取与预处理，实现参照 core/scared_dataset.py。
    仅当提供 list_filename 时使用该数据集，以复用另一程序中的预处理逻辑。
    """
    def __init__(self, aug_params=None, datapath=None, list_filename=None, training=True):
        super(SCAREDListDataset, self).__init__(aug_params, sparse=True)
        assert datapath is not None and os.path.exists(datapath), f"Invalid datapath: {datapath}"
        assert list_filename is not None and os.path.exists(list_filename), f"Invalid list file: {list_filename}"
        if read_all_lines is None or get_transform is None:
            raise RuntimeError("SCAREDListDataset requires data_io.get_transform/read_all_lines to be available.")

        self.datapath = datapath
        self.training = training
        # 从 aug_params 读取裁剪尺寸，默认与 config 一致：[H, W]
        if aug_params is not None and 'crop_size' in aug_params and aug_params['crop_size'] is not None:
            crop_size = aug_params['crop_size']
            # crop_size 形如 [H, W]
            self.crop_h, self.crop_w = int(crop_size[0]), int(crop_size[1])
        else:
            # 回退默认：与之前实现一致但以 HxW 的形式表达
            self.crop_h, self.crop_w = 320, 736

        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_files = [x[0] for x in splits]
        self.right_files = [x[1] for x in splits]
        self.disp_files = [x[2] for x in splits] if len(splits[0]) >= 3 else [None] * len(self.left_files)

        # 记录路径信息用于 __getitem__ 返回
        self.image_list = [[os.path.join(self.datapath, l), os.path.join(self.datapath, r)]
                           for l, r in zip(self.left_files, self.right_files)]
        self.disparity_list = [os.path.join(self.datapath, d) if d is not None else '' for d in self.disp_files]

    def __len__(self):
        return len(self.left_files)

    @staticmethod
    def _load_image(filename):
        return Image.open(filename).convert('RGB')

    @staticmethod
    def _load_disp(filename):
        # 与参考实现一致：/128. 将原始PNG存储的视差值缩放回浮点
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 128.0
        return data

    def __getitem__(self, index):
        left_path = os.path.join(self.datapath, self.left_files[index])
        right_path = os.path.join(self.datapath, self.right_files[index])
        disp_path = self.disp_files[index]
        disp_path_full = os.path.join(self.datapath, disp_path) if disp_path is not None else ''

        left_img = self._load_image(left_path)
        right_img = self._load_image(right_path)
        disparity = self._load_disp(disp_path_full) if (self.training and disp_path is not None) else \
                    (self._load_disp(disp_path_full) if disp_path is not None else None)

        if self.training:
            w, h = left_img.size
            crop_h, crop_w = self.crop_h, self.crop_w
            if w < crop_w or h < crop_h:
                # 简单兜底：缩放到至少裁剪尺寸
                left_img = left_img.resize((max(w, crop_w), max(h, crop_h)))
                right_img = right_img.resize((max(w, crop_w), max(h, crop_h)))
                if disparity is not None:
                    disparity = Image.fromarray(disparity)
                    disparity = np.array(disparity.resize((max(w, crop_w), max(h, crop_h)), Image.NEAREST), dtype=np.float32)
                w, h = left_img.size

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if disparity is not None:
                disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            processed = get_transform()
            left_t = processed(left_img)  # C,H,W torch.tensor
            right_t = processed(right_img)

            if disparity is None:
                disp_t = np.zeros((crop_h, crop_w), dtype=np.float32)
            else:
                disp_t = disparity.astype(np.float32)

            flow = np.stack([disp_t, np.zeros_like(disp_t)], axis=0)  # 2,H,W
            flow = torch.from_numpy(flow).float()
            valid = torch.from_numpy((disp_t > 0).astype(np.float32))  # H,W

            return [left_path, right_path, disp_path_full], left_t.float(), right_t.float(), flow[:1], valid
        else:
            # 验证/测试：按参考实现进行 padding 到 1024x1280
            w, h = left_img.size
            processed = get_transform()
            left_np = processed(left_img).numpy()   # C,H,W
            right_np = processed(right_img).numpy()

            target_height = 1024
            target_width = 1280
            top_pad = max(target_height - h, 0)
            right_pad = max(target_width - w, 0)

            left_np = np.lib.pad(left_np, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_np = np.lib.pad(right_np, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            left_t = torch.from_numpy(left_np).float()
            right_t = torch.from_numpy(right_np).float()

            if disparity is None:
                disp_t = np.zeros((left_t.shape[1], left_t.shape[2]), dtype=np.float32)
            else:
                disp_t = disparity.astype(np.float32)

            flow = np.stack([disp_t, np.zeros_like(disp_t)], axis=0)  # 2,H,W
            flow = torch.from_numpy(flow).float()
            valid = torch.from_numpy((disp_t > 0).astype(np.float32))

            return [left_path, right_path, disp_path_full], left_t, right_t, flow[:1], valid


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    # print('args.img_gamma', args.img_gamma)
    aug_params = {'crop_size': list(args.image_size), 'min_scale': args.spatial_scale[0],
                  'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = list(args.saturation_range)
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    print('train_datasets', args.train_datasets)
    for dataset_name in args.train_datasets:
        if dataset_name == 'sceneflow':
            new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'kitti':
            new_dataset = KITTI(aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params) * 140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params) * 5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name == 'eth3d_finetune':
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
            eth3d = ETH3D(aug_params)
            logging.info(f"Adding {len(eth3d)} samples from ETH3D")
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
        elif dataset_name == 'middlebury_train':
            tartanair = TartanAir(aug_params)
            logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
            sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
            fallingthings = FallingThings(aug_params)
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            carla = CARLA(aug_params)
            logging.info(f"Adding {len(carla)} samples from CARLA")
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            mb2005 = Middlebury(aug_params, split='2005')
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
            mb2006 = Middlebury(aug_params, split='2006')
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
            mb2014 = Middlebury(aug_params, split='2014')
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
            mb2021 = Middlebury(aug_params, split='2021')
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
            mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + carla * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
        elif dataset_name == 'middlebury_finetune':
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            carla = CARLA(aug_params)
            logging.info(f"Adding {len(carla)} samples from CARLA")
            mb2005 = Middlebury(aug_params, split='2005')
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
            mb2006 = Middlebury(aug_params, split='2006')
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
            mb2014 = Middlebury(aug_params, split='2014')
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
            mb2021 = Middlebury(aug_params, split='2021')
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
            mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            fallingthings = FallingThings(aug_params)
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            new_dataset = crestereo + instereo2k * 50 + carla * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 400 + fallingthings * 5
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
        elif dataset_name == 'scared':
            # 根据是否有split参数来决定使用哪个分割
            split = getattr(args, 'scared_split', 'train')  # 默认使用训练集
            new_dataset = SCARED(aug_params, split=split)
            logging.info(f"Adding {len(new_dataset)} samples from SCARED ({split} split)")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    return train_dataset


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2


    def gray_2_colormap_np(img, cmap='rainbow', max=None):
        img = img.cpu().detach().numpy().squeeze()
        assert img.ndim == 2
        img[img < 0] = 0
        mask_invalid = img < 1e-10
        if max == None:
            img = img / (img.max() + 1e-8)
        else:
            img = img / (max + 1e-8)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
        cmap_m = matplotlib.cm.get_cmap(cmap)
        map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
        colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
        colormap[mask_invalid] = 0

        return colormap


    def viz_disp(disp, scale=1, COLORMAP=cv2.COLORMAP_JET):
        disp_np = (torch.abs(disp[0].squeeze())).data.cpu().numpy()
        disp_np = (disp_np * scale).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_np, COLORMAP)
        return disp_color


    plot_dir = './temp/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    dataset = SceneFlowDatasets()

    for i in range(5):
        _, *data_blob = dataset[i]
        # 打印每个元素的形状以便调试
        for j, item in enumerate(data_blob):
            print(f"Item {j} shape: {item.shape if hasattr(item, 'shape') else 'No shape'}")

        image1, image2, disp_gt, valid = [x[None] for x in data_blob]
        print(f"Image1_oright shape: {image1.shape}")
        # print(f"Image2 shape: {image2.shape}")
        # print(f"Disp_gt shape: {disp_gt.shape}")
        # print(f"Valid shape: {valid.shape}")
        image1_np = image1[0].squeeze().cpu().numpy()
        print(f"Image1_after_squeeze shape: {image1.shape}")

        image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
        image1_np = image1_np.astype(np.uint8)

        disp_color = viz_disp(disp_gt, scale=5)
        cv2.imwrite(os.path.join(plot_dir, f'{i}_disp_gt.png'), disp_color)

        disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
        cv2.imwrite(os.path.join(plot_dir, f'{i}_disp_gt1.png'), disp_gt_np[:, :, ::-1])
        print(f"Image1 shape: {image1.shape}")
        image1 = image1[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        print(f"Image1_after shape: {image1.shape}")
        cv2.imwrite(os.path.join(plot_dir, f'{i}_img1.png'), image1)

# Item 0 shape: torch.Size([3, 540, 960])
# Item 1 shape: torch.Size([3, 540, 960])
# Item 2 shape: torch.Size([1, 540, 960])
# Item 3 shape: torch.Size([540, 960])
# Image1 shape: torch.Size([1, 3, 540, 960])
# Image2 shape: torch.Size([1, 3, 540, 960])
# Disp_gt shape: torch.Size([1, 1, 540, 960])
# Valid shape: torch.Size([1, 540, 960])
# Image1 shape: torch.Size([1, 3, 540, 960])
# Image1_after shape: (540, 960, 3)



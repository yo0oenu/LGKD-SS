import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class camvidDataset(CustomDataset):
    CLASSES = (
        'Building', 'Bicyclist', 'Fence', 'Pole', 'Pedestrian', 'Road',
        'Sky', 'SignSymbol', 'Tree', 'Sidewalk', 'Car'
    )

    PALETTE = [
        [128, 0, 0],      # 0: Building
        [0, 128, 192],    # 1: Bicyclist
        [64, 64, 128],    # 2: Fence
        [192, 192, 128],  # 3: Pole
        [64, 64, 0],      # 4: Pedestrian
        [128, 64, 128],   # 5: Road
        [128, 128, 128],  # 6: Sky
        [192, 128, 128],  # 7: SignSymbol
        [128, 128, 0],    # 8: Tree
        [0, 0, 192],      # 9: Sidewalk
        [64, 0, 128]      # 10: Car
    ]

    def results2img(self, results, imgfile_prefix, to_label_id=False):
        """Write the segmentation results to images with CamVid palette."""
        result_files = []
        for idx, result in enumerate(results):
            img_info = self.img_infos[idx]
            basename = osp.splitext(osp.basename(img_info['filename']))[0]
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')
            output = result
            if hasattr(output, 'shape') and output.ndim == 3:
                output = output.squeeze(0)
            output = output.astype(np.uint8)
            # 팔레트 적용
            output_img = Image.fromarray(output)
            output_img.putpalette(np.array(self.PALETTE, dtype=np.uint8).flatten())
            output_img.save(png_filename)
            result_files.append(png_filename)
        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=False):
        """Format the results for evaluation or submission."""
        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)
        return result_files, tmp_dir

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the results using mIoU and aAcc."""
        eval_results = super().evaluate(results, metric=metric, logger=logger, **kwargs)
        return eval_results
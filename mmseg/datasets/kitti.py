import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class kittiDataset(CustomDataset):
    CLASSES = (
        'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 
        'pole', 'car', 'sign', 'pedestrian', 'cyclist'
    )

    PALETTE = [
        [128, 128, 128],   # 0: sky
        [128, 0, 0],       # 1: building
        [128, 64, 128],    # 2: road
        [0, 0, 192],       # 3: sidewalk
        [64, 64, 128],     # 4: fence
        [128, 128, 0],     # 5: vegetation
        [192, 192, 128],   # 6: pole
        [64, 0, 128],      # 7 : car
        [192, 128, 128],   # 8: sign
        [64, 64, 0],       # 9: pedestrian
        [0, 128, 192]      # 10: cyclist
]

    def results2img(self, results, imgfile_prefix, to_label_id=False):
        result_files = []
        for idx, result in enumerate(results):
            img_info = self.img_infos[idx]
            basename = osp.splitext(osp.basename(img_info['filename']))[0]
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')
            result_files.append(png_filename)
            output = result
            if hasattr(output, 'shape') and output.ndim == 3:
                output = output.squeeze(0)
            output = output.astype(np.uint8)
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
import os
import numpy as np
from pycocotools.coco import COCO
# 核心：导入 DATASET 注册器（解决未解析问题）
from alphapose.models.builder import DATASET
# 导入 AlphaPose 内置的 bbox 裁剪工具（替代手动裁剪逻辑，对齐官方规范）
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
# 导入自定义数据集基类
from .custom import CustomDataset


@DATASET.register_module
class CrowdPose(CustomDataset):
    CLASSES = ['person']
    EVAL_JOINTS = list(range(14))
    num_joints = 14
    # CrowdPose关键点左右翻转配对
    joint_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]

    def _load_jsons(self):
        items = []
        labels = []
        _coco = self._lazy_load_ann_file()  # 替换为基类的懒加载方法（更高效）

        # 验证类别
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "类别不匹配"

        self.json_id_to_contiguous = {v: k for k, v in enumerate(_coco.getCatIds())}

        # 加载图像和标注
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            abs_path = os.path.join(self._root, self._img_prefix, entry['file_name'])
            if not os.path.exists(abs_path):
                raise IOError(f"图像不存在: {abs_path}")

            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue
            for obj in label:
                items.append(abs_path)
                labels.append(obj)
        return items, labels

    def _check_load_keypoints(self, coco, entry):
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        valid_objs = []
        width, height = entry['width'], entry['height']

        for obj in objs:
            # 过滤非人物类别
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                continue
            # 过滤无关键点的标注
            if max(obj['keypoints']) == 0 or obj.get('num_keypoints', 0) == 0:
                continue

            # 替换为 AlphaPose 官方 bbox 转换+裁剪工具（更健壮）
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # 过滤无效 bbox
            if (xmax - xmin) * (ymax - ymin) <= 0 or xmax <= xmin or ymax <= ymin:
                continue

            # 加载关键点（14个，格式：x,y,visibility）
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]  # x
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]  # y
                # 可见性：0=不可见，1=可见，2=标注但遮挡 → 统一为 0/1
                joints_3d[i, :2, 1] = 1 if obj['keypoints'][i * 3 + 2] > 0 else 0

            # 过滤无可见关键点的标注
            if np.sum(joints_3d[:, 0, 1]) < 1:
                continue

            # 对齐 AlphaPose 官方逻辑：训练时检查关键点与bbox中心匹配度
            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        # 若无有效标注，添加dummy标签（避免验证时COCO metric报错）
        if not valid_objs and not self._skip_empty:
            valid_objs.append({
                'bbox': np.array([-1, -1, 0, 0]),
                'width': width,
                'height': height,
                'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
            })
        return valid_objs

    # 补充基类依赖的 bbox 中心计算方法
    def _get_box_center_area(self, bbox):
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    # 补充基类依赖的关键点中心计算方法
    def _get_keypoints_center_count(self, keypoints):
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
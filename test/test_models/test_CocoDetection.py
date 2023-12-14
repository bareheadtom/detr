import unittest
import sys
sys.path.append('../../')
from datasets.coco import CocoDetection as WCOCO,ConvertCocoPolysToMask,convert_coco_poly_to_mask  # 请将 'your_module' 替换为存放 CocoDetection 类的模块名称
from PIL import Image
import torch
import torchvision
class TestCocoDetection(unittest.TestCase):
    # def setUp(self):
    #     self.img_folder = '/datasets/COCO/train2017'
    #     self.ann_file = '/datasets/COCO/annotations/instances_train2017.json'
    #     self.transforms = None  # 这里需要填写 transforms 函数
    #     self.return_masks = True  # 这里根据需要设置 return_masks 值

    #     self.coco_dataset = WCOCO(self.img_folder, self.ann_file, self.transforms, self.return_masks)


    def build_ancfg(self):
        boxes = torch.tensor([
            [1.0800, 187.6900, 612.6700, 473.5300],
            [311.7300, 4.3100, 631.0100, 232.9900],
        ])
        labels = torch.tensor([51, 56])
        # 创建示例 masks（这里使用全零张量表示示例）
        masks = torch.zeros((2, 480, 640), dtype=torch.uint8)  # 假设 masks 的形状为 (8, 100, 100)
        # 创建示例数据的其余部分
        image_id = torch.tensor([2])
        area = torch.tensor([120057.1406, 44434.7500])
        iscrowd = torch.tensor([0, 0])
        orig_size = torch.tensor([480, 640])
        size = torch.tensor([480, 640])
        # 构造字典数据
        anno = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': orig_size,
            'size': size
        }
        return anno
    def buildsupercfg(self):
        target = [
            {'segmentation': [[267.38, 330.14, 281.81, 314.75, 299.12, 282.05, 281.81, 258.96, 248.14, 224.34, 242.37, 189.71, 213.52, 186.83, 237.56, 160.86, 214.48, 156.05, 224.1, 134.89, 204.86, 128.16, 204.86, 109.89, 209.67, 84.88, 228.91, 79.11, 254.88, 66.6, 319.32, 59.87, 301.04, 31.02, 328.93, 50.25, 353.94, 35.83, 389.53, 49.29, 429.92, 74.3, 451.08, 79.11, 454.93, 90.65, 453.01, 99.31, 459.74, 112.77, 456.85, 132.01, 453.01, 149.32, 442.43, 170.48, 432.81, 178.17, 430.89, 204.14, 410.69, 180.1, 399.15, 218.57, 386.64, 242.61, 358.75, 266.66, 350.09, 289.74, 363.56, 312.82, 388.57, 330.14, 374.14, 347.45, 324.13, 355.14, 263.53, 339.76, 269.3, 329.18]], 'area': 47675.66289999999, 'iscrowd': 0, 'image_id': 30, 'bbox': [204.86, 31.02, 254.88, 324.12], 'category_id': 64, 'id': 291613}, 
            {'segmentation': [[394.34, 155.81, 403.96, 169.28, 403.96, 198.13, 389.53, 240.45, 364.52, 257.76, 352.98, 279.88, 348.17, 289.5, 354.9, 303.93, 372.22, 312.58, 383.76, 323.16, 372.22, 339.51, 327.01, 350.09, 296.23, 351.06, 270.27, 337.59, 267.38, 330.86, 266.42, 327.01, 292.39, 311.62, 295.27, 273.15, 280.84, 259.69, 255.84, 234.68, 237.56, 191.4, 260.65, 218.33, 285.65, 209.67, 319.32, 205.82, 330.86, 197.17, 338.55, 175.05, 349.13, 170.24, 375.1, 165.43, 370.29, 184.67, 376.06, 183.7, 381.83, 171.2, 389.53, 160.62, 393.38, 157.73]], 'area': 16202.798250000003, 'iscrowd': 0, 'image_id': 30, 'bbox': [237.56, 155.81, 166.4, 195.25], 'category_id': 86, 'id': 1155486}
            ]
        return target
    # def test_getitem(self):
    #     print("\n***************test_getitem")
    #     # 测试 __getitem__ 方法
    #     idx = 2  # 你可以根据需要设置索引
    #     img, target = self.coco_dataset[idx]

    #     print("after ConvertCocoPolysToMask img",img,"target",len(target),target)
    #     for k,v in target.items():
    #         print(k,v.shape)

    def test_CocoDetectionContent(self):
        contomask = ConvertCocoPolysToMask(True)
        print("\n***************test_ConvertCocoPolysToMask")
        image,target = Image.new('RGB', (480, 640), (0, 0, 0)),self.buildsupercfg()
        image_id = 30
        target = {'image_id':image_id,'annotations':target}
        image, target = contomask(image,target)
        print(type(image), type(target),target)
    def test_convert_coco_poly_to_mask(self):
        print("\n***************test_convert_coco_poly_to_mask")
        target = self.buildsupercfg()
        image_id = 30
        target = {'image_id':image_id,'annotations':target}
        #428, 640
        anno = target["annotations"]
        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations,428,640)
        print("masks",masks.shape,masks)
    def test_torchvisiondatasetsCocoDetection(self):
        print("\n***************test_torchvisiondatasetsCocoDetection")
        cocodetec = torchvision.datasets.CocoDetection('/datasets/COCO/train2017','/datasets/COCO/annotations/instances_train2017.json')
        image, target = cocodetec[2]
        print(type(image),"target",target)
        #print("ids",cocodetec.ids)
        image = cocodetec._load_image(cocodetec.ids[2])
        target = cocodetec._load_target(cocodetec.ids[2])
        print(type(image),"target",target)


if __name__ == '__main__':
    unittest.main()

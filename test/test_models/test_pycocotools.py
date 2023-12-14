import unittest
import sys
sys.path.append('../../')
from datasets.coco import CocoDetection as WCOCO,ConvertCocoPolysToMask,convert_coco_poly_to_mask  # 请将 'your_module' 替换为存放 CocoDetection 类的模块名称
from PIL import Image
import torch
import torchvision
from pycocotools.coco import COCO
class TestPycocoTools(unittest.TestCase):
    def setUp(self):
        self.coco = COCO( annotation_file='/datasets/COCO/annotations/instances_train2017.json')
        #print("dataset",self.coco.dataset)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    
    def test_print(self):
         #print("dataset",self.coco.dataset)
        print("\n************test_print")
        print("info",self.coco.dataset['info'])
        print("licenses",self.coco.dataset['licenses'][:2])
        print("images",self.coco.dataset['images'][:2])
        print("annotations",self.coco.dataset['annotations'][:2])
        print("categories",self.coco.dataset['categories'][:2])
    
    
    def test_loadImgs(self):
        print("\n************test_loadImgs")
        images = self.coco.loadImgs(self.ids[2])
        print(images)
    
    def test_loadAnns(self):
        print("\n************test_loadAnns")
        anns = self.coco.loadAnns(self.coco.getAnnIds(self.ids[2]))
        print(anns)







if __name__ == '__main__':
    unittest.main()
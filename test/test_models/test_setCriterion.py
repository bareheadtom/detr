import unittest
import torch
import sys
sys.path.append('../../')
from models.detr import SetCriterion

# 导入你的类
from models.matcher import HungarianMatcher

class TestsetCriterion(unittest.TestCase):
    def setUp(self):
        matcher =HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2
        # TODO this is a hack
        if True:
            aux_weight_dict = {}
            for i in range(6 - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(91, matcher=matcher, weight_dict=weight_dict,
                                eos_coef=0.1, losses=losses)
        
        self.outputs = {
            "pred_logits": torch.randn(2, 3, 92),  # 示例形状，可以根据你的实际数据进行替换
            "pred_boxes": torch.tensor([
                [[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]],
                [[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.1,0.2,0.3,0.4]]
                ])
        }

        self.targets = [
            {"labels": torch.tensor([1, 2, 3]), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]])},
            {"labels": torch.tensor([2, 3]), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8]])}
        ]
        self.indices = matcher(self.outputs, self.targets)
        #print("indices",self.indices)
        #indices [(tensor([0, 1, 2]), tensor([0, 1, 2])), (tensor([0, 1]), tensor([0, 1]))]

    def test_forward(self):
        print("\n*******test_forward")
        losses = self.criterion.forward(self.outputs, self.targets)
        print("total losses", losses)
    
    def test_loss_labels(self):
        print("\n*******test_loss_labels")
        loss_ce = self.criterion.loss_labels(self.outputs, self.targets,self.indices ,5,True)
        print("loss_ce",loss_ce)
    
    def test_loss_cardinality(self):
        print("\n*******test_loss_cardinality")
        #def loss_cardinality(self, outputs, targets, indices, num_boxes):
        cardinality_error = self.criterion.loss_cardinality(self.outputs, self.targets, self.indices, 5)
        print("cardinality_error",cardinality_error)

    def test_loss_boxes(self):
        print("\n*******test_loss_boxes")
        bbox_giou = self.criterion.loss_boxes(self.outputs, self.targets, self.indices, 5)
        print("bbox_giou",bbox_giou)
    
    def test_get_src_permutation_idx(self):
        print("\n*******test_get_src_permutation_idx")
        print("indices",self.indices)
        #src_idx = self.criterion._get_src_permutation_idx(self.indices)
        #print("src_idx",src_idx)
        # for i, (src, _) in enumerate(self.indices):
        #     print("i",i,"(src, _)",(src, _))
        #     print("torch.full_like(src, i)",torch.full_like(src, i))
        # print("[torch.full_like(src, i) for i, (src, _) in enumerate(self.indices)]",[torch.full_like(src, i) for i, (src, _) in enumerate(self.indices)])

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(self.indices)])
        src_idx = torch.cat([src for (src, _) in self.indices])
        print("batch_idx",batch_idx,"src_idx",src_idx)

    def test__get_tgt_permutation_idx(self):
        print("\n*******test__get_tgt_permutation_idx")
        print("indices",self.indices)
        tgt_idx = self.criterion._get_tgt_permutation_idx(self.indices)
        print("tgt_idx",tgt_idx)
    
    # def test_loss_masks(self):
    #     print("\n*******test_loss_masks")
    #     mask_dice = self.criterion.loss_masks(self.outputs, self.targets, self.indices, 5)
    #     print("mask_dice",mask_dice)


    


if __name__ == '__main__':
    unittest.main()



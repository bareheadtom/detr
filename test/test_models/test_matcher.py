import unittest
import torch
import sys
sys.path.append('../../')

# 导入你的类
from models.matcher import HungarianMatcher

class TestHungarianMatcher(unittest.TestCase):
    def test_HungarianMatcher(self):
        print("\n***test_HungarianMatcher")
        # 创建一个匹配器实例
        matcher = HungarianMatcher()

        # 模拟输入数据
        outputs = {
            "pred_logits": torch.randn(2, 3, 10),  # 示例形状，可以根据你的实际数据进行替换
            "pred_boxes": torch.tensor([
                [[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]],
                [[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.1,0.2,0.3,0.4]]
                ])
        }

        targets = [
            {"labels": torch.tensor([1, 2, 3]), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]])},
            {"labels": torch.tensor([2, 3]), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8]])}
        ]

        # 执行 forward 方法
        result = matcher(outputs, targets)
        print(result)
        # [(tensor([0, 1, 2]), tensor([2, 1, 0])), (tensor([1, 2]), tensor([1, 0]))]

if __name__ == '__main__':
    unittest.main()

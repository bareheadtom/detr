import unittest
import sys
sys.path.append('../../')
from datasets.coco import build as build_coco
import util.misc as utils

import torch
from torch.utils.data import DataLoader, DistributedSampler


class TestBuildDataset(unittest.TestCase):
    def test_build_dataset_coco(self):
        # 模拟参数
        class Args:
            coco_path = '/datasets/COCO/'
            masks = False

        args = Args()

        # 调用函数并断言预期的返回结果
        dataset_train = build_coco('train', args)

    def test_data_loader(self):
        device = torch.device('cuda')
        class Args:
            coco_path = '/datasets/COCO/'
            masks = False
            batch_size = 2
            num_workers = 2

        args = Args()

        # 调用函数并断言预期的返回结果
        dataset_train = build_coco('train', args)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
        print("data_loader_train",data_loader_train)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(1)
        print_freq = 10

        for samples, targets in data_loader_train:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            tensors, mask = samples.decompose()
            print("tensors.shape:",tensors.shape, "maske.shape", mask.shape)
            #print("targets",targets)
        



if __name__ == '__main__':
    unittest.main()
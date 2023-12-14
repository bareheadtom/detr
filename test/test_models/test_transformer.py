import torch
import unittest
import sys
from torch import nn, Tensor
sys.path.append('../../')
from models.transformer import TransformerEncoderLayer, TransformerEncoder,TransformerDecoderLayer,TransformerDecoder, Transformer
  # 请将"your_module"替换为实际定义TransformerEncoderLayer的模块或文件名

class TestTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        # 在每个测试用例之前设置
        self.d_model = 512
        self.nhead = 8
        self.singleEncoder = TransformerEncoderLayer(512, 8, dim_feedforward=1024)

    def test_forward_post(self):
        print("\n***test_forward_post")
        src = torch.randn(2, 10, 512)  # 示例输入数据
        output = self.singleEncoder.forward_post(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)

    def test_forward_pre(self):
        print("\n***test_forward_pre")
        src = torch.randn(2, 10, 512)  # 示例输入数据
        output = self.singleEncoder.forward_pre(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)

    def test_forward(self):
        print("\n***test_forward")
        src = torch.randn(2, 10, 512)  # 示例输入数据
        output = self.singleEncoder.forward(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)
    
    def test_nn_MultiheadAttention(self):
        print("\n***test_nn_MultiheadAttention")
        input_dim = 128
        seq_length = 10
        num_heads = 4

        # 创建一个 nn.MultiheadAttention 实例
        multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # 假设有一个输入序列input_seq，形状为 (sequence_length, batch_size, input_dim)
        input_seq = torch.randn(seq_length, 2, input_dim)  # 假设 batch_size 为 1

        # 假设query、key、value都使用相同的输入序列input_seq
        # 注意：在实际应用中，query、key、value可以是不同的张量
        query = torch.randn(seq_length//2, 2, input_dim)
        key = torch.randn(seq_length, 2, input_dim)
        value = torch.randn(seq_length, 2, input_dim)

        # 进行多头注意力计算
        # 注意：在实际应用中，你可能需要对输入数据进行更多的预处理和填充操作
        attn_output, attn_weights = multihead_attn(query, key, value)

        # attn_output是注意力加权后的输出结果，形状为 (sequence_length, batch_size, embed_dim)
        # attn_weights包含了注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)

        print("注意力加权输出的形状:", attn_output.shape)
        print("注意力权重的形状:", attn_weights.shape)
    
    def test_transformerEncoder(self):
        print("\n***test_transformerEncoder")
        src = torch.randn(2, 10, 512)
        model = TransformerEncoder(self.singleEncoder, 6)
        output = model(src)
        print(output.shape)
        self.assertEqual(output.shape, src.shape)
    
    def test_TransformerDecoderLayer(self):
        print("\n***test_TransformerDecoderLayer")
        src = torch.randn(2, 10, 512)
        tgt = torch.randn(2, 10, 512)  # 示例输入数据
        encoder = TransformerEncoder(self.singleEncoder, 6)
        memmory = encoder(src)
        mask = torch.randn(10,2)
        decoderlayer = TransformerDecoderLayer(512, 8, dim_feedforward=2048,normalize_before=False)
        out = decoderlayer(tgt, memmory,memory_key_padding_mask = mask)
        print(out.shape)
        self.assertEqual(out.shape, src.shape)
    
    def test_TransformerDecoder(self):
        print("\n***test_TransformerDecoder")
        src = torch.randn(2, 10, 512)
        tgt = torch.randn(2, 10, 512)  # 示例输入数据
        norm = nn.LayerNorm(512)
        encoder = TransformerEncoder(self.singleEncoder, 6)
        memmory = encoder(src)
        decoderlayer = TransformerDecoderLayer(512, 8, dim_feedforward=2048,normalize_before=False)
        decoder = TransformerDecoder(decoderlayer, 6, norm=norm,return_intermediate=True)
        out = decoder(tgt, memmory)
        print(out.shape)
        #self.assertEqual(out.shape, src.shape)

    def test_Transformer(self):
        print("\n***test_Transformer")
        target_dim = 100
        input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        feature = torch.randn(2, 2048, 23, 35)

        print("feature.flatten(2).shape",feature.flatten(2).shape)
        print("feature.permute(2, 0, 1).shape",feature.flatten(2).permute(2, 0, 1).shape)

        pos_embed = torch.randn(2, 256, 23, 35)
        mask = torch.randn(2, 23, 35)
        query_embed = torch.randn(target_dim, 256)#pos and tgt
        print("query_embed",query_embed.unsqueeze(1).shape)
        query_embed2 = query_embed.unsqueeze(1).repeat(1, 2, 1).shape
        print("query_embed2",query_embed2)
        transformerModel = Transformer(d_model=256,nhead=8,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=1024)
        
        out = transformerModel(input_proj(feature), mask, query_embed, pos_embed)[0]
        print("out.shape",out.shape)



       


if __name__ == '__main__':
    unittest.main()

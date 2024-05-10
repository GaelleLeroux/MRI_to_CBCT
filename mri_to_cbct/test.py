from nets.cut_G import Generator
from nets.cut_P import Head
from nets.cut_D import Discriminator
import torch


G = Generator()
D_Y = Discriminator()
H = Head()


x = torch.randn((1, 1, 256, 256,256)).to()
print(x.shape)
G = Generator().to()
feat_k_pool, sample_ids = G(x, encode_only=True, patch_ids=None)
feat_q_pool, _ = G(x, encode_only=True, patch_ids=sample_ids)
print(len(feat_k_pool))


# print("oui")
# G(torch.rand(1, 1, 128, 128, 128))
# print("fi")
# D_Y(torch.rand(1, 1, 256, 256, 256))
# tgt = torch.rand(1, 1, 256, 256, 256)
# src = torch.rand(1, 1, 256, 256, 256)
# feat_q, patch_ids_q = G(tgt, encode_only=True)
# feat_k, _ = G(src, encode_only=True, patch_ids=patch_ids_q)

# feat_k_pool =H(feat_k)
# feat_q_pool = H(feat_q)
# H(torch.rand(1, 1, 128, 128, 128))
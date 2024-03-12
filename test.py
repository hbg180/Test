import timm
import torch.nn as nn

svt = timm.create_model('twins_svt_small', pretrained=False)
# del svt.head
# del svt.patch_embeds[2]
# del svt.patch_embeds[2]
# del svt.blocks[2]
# del svt.blocks[2]
# del svt.pos_block[2]
# del svt.pos_block[2]
# del svt.pos_block[1]
print(svt)
# nn.LayerNorm()

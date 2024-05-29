import torch
import torch.nn as nn

rnn = nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
output2, _ = rnn(input)
hn2, __ = rnn(input)

print(output.size())
print(hn.size())
# print(output2.size())
# print(hn2.size())
# print(_.size())
# print(__.size())
# print(output == output2)

rnn = nn.GRU(32, 32, 3, batch_first=True)
input = torch.randn(1, 1024, 32)
h0 = torch.randn(3, 1024, 32)
output, hn = rnn(input, h0)
print(rnn)
print(output.size())
print(hn.size())
# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)
#
# print(src.size())
# print(tgt.size())
# print(out.size())
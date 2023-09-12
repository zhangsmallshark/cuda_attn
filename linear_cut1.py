import math
from torch import nn
from torch.autograd import Function
import torch

import linear_cutlass

torch.manual_seed(42)


class LinearCutFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        outputs = linear_cutlass.forward(input, weights, bias)
        output = outputs[0]
        print('in linear cut')
        print(output.shape)
        variables = [input, weights]
        ctx.save_for_backward(*variables)
        refer_out = torch.mm(input, weights.transpose())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        outputs = linear_cutlass.backward(
            grad_output.contiguous() *ctx.saved_variables)
        d_input, d_weights, d_bias = outputs
        return d_input, d_weights, d_bias


class LinearCut(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearCut, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return LinearCutFunction.apply(input, self.weights, self.bias)
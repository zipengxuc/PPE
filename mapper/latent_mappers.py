import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from stylegan import EqualLinear, PixelNorm, StyleGANv2Generator

class Mapper(nn.Layer):

    def __init__(self):
        super().__init__()
        self.mapping = [PixelNorm()]

        for _ in range(4):
            self.mapping.append(
                EqualLinear(
                    512, 512, lr_mul=0.01, activation='fused_lrelu'
                )
            )
        self.mapping = nn.Sequential(*self.mapping)

    def forward(self, x):
        x = self.mapping(x)
        return x

class SingleMapper(nn.Layer):

    def __init__(self):
        super().__init__()
        self.mapping = Mapper()

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(nn.Layer):

    def __init__(self):
        super().__init__()

        self.course_mapping = Mapper()
        self.medium_mapping = Mapper()
        self.fine_mapping = Mapper()

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)

        out = paddle.concat([x_coarse, x_medium, x_fine], axis=1)
        return out
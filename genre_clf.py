from typing import Union, Tuple
from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Linear, \
    Sequential, Dropout2d, Softmax, LazyLinear, Dropout


class GenreClf(Module):

    def __init__(self,
                 cnn_channels_out_1: int,
                 cnn_kernel_1: Union[Tuple[int, int], int],
                 cnn_stride_1: Union[Tuple[int, int], int],
                 cnn_padding_1: Union[Tuple[int, int], int],
                 pooling_kernel_1: Union[Tuple[int, int], int],
                 pooling_stride_1: Union[Tuple[int, int], int],
                 cnn_channels_out_2: int,
                 cnn_kernel_2: Union[Tuple[int, int], int],
                 cnn_stride_2: Union[Tuple[int, int], int],
                 cnn_padding_2: Union[Tuple[int, int], int],
                 pooling_kernel_2: Union[Tuple[int, int], int],
                 pooling_stride_2: Union[Tuple[int, int], int],
                 cnn_channels_out_3: int,
                 cnn_kernel_3: Union[Tuple[int, int], int],
                 cnn_stride_3: Union[Tuple[int, int], int],
                 cnn_padding_3: Union[Tuple[int, int], int],
                 pooling_kernel_3: Union[Tuple[int, int], int],
                 pooling_stride_3: Union[Tuple[int, int], int],
                 fc_out_1: int,
                 clf_output_classes: int,
                 dropout_conv_1: float,
                 dropout_conv_2: float,
                 dropout_conv_3: float,
                 dropout_fc_1: float,
                 ) -> None:

        super().__init__()

        self.conv1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=cnn_channels_out_1,
                   kernel_size=cnn_kernel_1,
                   stride=cnn_stride_1,
                   padding=cnn_padding_1),
            ReLU(),
            BatchNorm2d(cnn_channels_out_1),
            MaxPool2d(kernel_size=pooling_kernel_1,
                      stride=pooling_stride_1),
            Dropout2d(dropout_conv_1)
        )

        self.conv2 = Sequential(
            Conv2d(in_channels=cnn_channels_out_1,
                   out_channels=cnn_channels_out_2,
                   kernel_size=cnn_kernel_2,
                   stride=cnn_stride_2,
                   padding=cnn_padding_2),
            ReLU(),
            BatchNorm2d(cnn_channels_out_2),
            MaxPool2d(kernel_size=pooling_kernel_2,
                      stride=pooling_stride_2),
            Dropout2d(dropout_conv_2)
        )

        self.conv3 = Sequential(
            Conv2d(in_channels=cnn_channels_out_2,
                   out_channels=cnn_channels_out_3,
                   kernel_size=cnn_kernel_3,
                   stride=cnn_stride_3,
                   padding=cnn_padding_3),
            ReLU(),
            BatchNorm2d(cnn_channels_out_3),
            MaxPool2d(kernel_size=pooling_kernel_3,
                      stride=pooling_stride_3),
            Dropout2d(dropout_conv_3)
        )

        self.fc1 = Sequential(LazyLinear(out_features=fc_out_1),
                              ReLU(),
                              Dropout(dropout_fc_1))

        self.clf = Sequential(Linear(in_features=fc_out_1, out_features=clf_output_classes),
                              Softmax(dim=1))

    def forward(self,
                x: Tensor) -> Tensor:
        h = x if x.ndimension() == 4 else x.unsqueeze(1)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = h.view(h.size()[0], -1)
        h = self.fc1(h)
        return self.clf(h)


if __name__ == '__main__':
    pass

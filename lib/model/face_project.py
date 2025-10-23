import torch.nn as nn
import torch


def generate_hidden_dims(input_dim, output_dim, num_layers=4, factor=2):
    hidden_dims = []
    current_dim = input_dim

    for _ in range(num_layers):
        current_dim = max(output_dim, current_dim // factor)  # 不能比 output_dim 小
        hidden_dims.append(current_dim)

    return hidden_dims


def generate_hidden_dims_linear(input_dim, output_dim, num_layers=4):
    step = (input_dim - output_dim) // num_layers
    hidden_dims = [input_dim - step * (i + 1) for i in range(num_layers)]
    return hidden_dims


class FaceToTextEmbeddingMapping(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4, method="geo"):
        super(FaceToTextEmbeddingMapping, self).__init__()

        # 自动生成 hidden_dims
        if method == "geo":
            hidden_dims = generate_hidden_dims(input_dim, output_dim, num_layers)
        else:
            hidden_dims = generate_hidden_dims_linear(input_dim, output_dim, num_layers)

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        return self.fc_out(x)


# class FaceToTextEmbeddingMapping(nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers=4, method="geo"):
#         super(FaceToTextEmbeddingMapping, self).__init__()

#         # 自动生成 hidden_dims
#         if method == "geo":
#             hidden_dims = generate_hidden_dims(input_dim, output_dim, num_layers)
#         else:
#             hidden_dims = generate_hidden_dims_linear(input_dim, output_dim, num_layers)

#         self.layers = nn.ModuleList()
#         prev_dim = input_dim

#         for hidden_dim in hidden_dims:
#             self.layers.append(nn.Linear(prev_dim, hidden_dim))
#             # self.layers.append(nn.BatchNorm1d(hidden_dim))
#             self.layers.append(nn.LeakyReLU(negative_slope=0.2))
#             # self.layers.append(nn.Dropout(0.3))
#             prev_dim = hidden_dim

#         self.fc_out = nn.Linear(prev_dim, output_dim)

#     def forward(self, x):
#         # ⚠️ 在forward中临时开启确定性算法
#         prev_state = torch.are_deterministic_algorithms_enabled()
#         torch.use_deterministic_algorithms(True)
#         try:
#             for layer in self.layers:
#                 x = layer(x)
#             x = self.fc_out(x)
#         finally:
#             # 恢复之前的全局状态，避免影响外部代码
#             torch.use_deterministic_algorithms(prev_state)
#         return x
    

class CLIPToTextEmbeddingMapping(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4, method="geo"):
        super(CLIPToTextEmbeddingMapping, self).__init__()

        # 自动生成 hidden_dims
        if method == "geo":
            hidden_dims = generate_hidden_dims(input_dim, output_dim, num_layers)
        else:
            hidden_dims = generate_hidden_dims_linear(input_dim, output_dim, num_layers)

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim)) # 去掉 BatchNorm 和 Dropout 获取确定性输出
            self.layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # ⚠️ 在forward中临时开启确定性算法
        prev_state = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for layer in self.layers:
                x = layer(x)
            x = self.fc_out(x)
        finally:
            # 恢复之前的全局状态，避免影响外部代码
            torch.use_deterministic_algorithms(prev_state)
        return x
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class AngularAggLayer(nn.Module):
    def __init__(self, num_nodes, num_class, input_dim, labels, dropout, device, name):
        super(AngularAggLayer, self).__init__()
        # 储存计算出的mean_tensor
        self.register_buffer("mean_tensor", torch.randn(num_nodes, num_class))
        # 储存输入特征的幅值
        self.register_buffer("mag", torch.randn(1, input_dim))
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.labels = labels
        self.dropout = dropout
        num = int(num_class * (num_class - 1) / 2)
        self.theta = torch.nn.Parameter(torch.randn(num, ) * torch.pi)
        self.params = torch.nn.Parameter(torch.randn(2 * input_dim, 1, dtype=torch.complex64))
        self.device = device
        self.name = name

    def get_theta(self):
        return self.theta


    def post_norm(self, features):
        # 计算输入特征的幅值
        features_norm = torch.sqrt(features.real ** 2 + features.imag ** 2).to(self.device)
        # 输入特征/模（归一化）
        unit_norm_result = features / features_norm
        # 对幅度进行调整
        normalized_result = unit_norm_result * self.mag
        return normalized_result

    def negative_symmetric_theta(self):
        triu_indices = torch.triu_indices(self.num_class, self.num_class, offset=1)  # 上三角
        tril_indices = torch.tril_indices(self.num_class, self.num_class, offset=-1)  # 下三角

        # 构造对称矩阵
        matrix = torch.zeros(self.num_class, self.num_class).to(self.device)
        matrix[triu_indices[0], triu_indices[1]] = self.theta
        matrix[tril_indices[1], tril_indices[0]] = -self.theta

        return matrix

    def get_centers(self, features, labels):
        # 将样本标签转换为one-hot编码
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_class).to(self.device)
        complex_one_hot = one_hot.t().to(torch.complex64)
        #  计算每个class内fearture vector之和
        sum_by_label = torch.matmul(complex_one_hot, features)
        #  计算每个class的样本数量
        count_by_label = one_hot.sum(dim=0).unsqueeze(1)
        #  计算centers（类别中心）
        mean_tensor = sum_by_label / count_by_label
        return mean_tensor

    def generate_matrix(self, weight, label):
        label_indices = label.long()
        # 获取class对应的weights row
        A_hat = torch.index_select(weight, 0, label_indices).to(self.device)
        # 获取并增加class对应的weights column，得到完整的class对应的weights
        A_hat = torch.index_select(A_hat, 1, label).to(self.device)
        return A_hat

    def get_label(self, feature):
        # 连接input features和mean_tensors
        concatenated = torch.cat([feature.unsqueeze(1).expand(-1, self.mean_tensor.size(0), -1),
                                  self.mean_tensor.unsqueeze(0).expand(feature.size(0), -1, -1)], dim=2)
        # 计算得分矩阵
        scores = torch.matmul(concatenated, self.params)
        scores_abs = torch.sqrt(scores.real ** 2 + scores.imag ** 2)
        # index得分矩阵第一个division的max，得到的index即为每个样本对应的center
        _, nearest_center_idx = scores_abs.max(dim=1)
        return nearest_center_idx.view(-1).squeeze()

    def forward(self, x, A, l0=False, writer=None, epoch=0):
        # 计算幅值
        modulus = torch.abs(x).to(self.device) + 1e-5
        # 归一化
        norm_features = x / modulus
        # 计算并保存类别中心
        self.mean_tensor = self.get_centers(norm_features, self.labels)
        matrix = self.negative_symmetric_theta()
        fake_label = self.get_label(norm_features)
        # 生成根据fake label调整后的metrix
        A_hat = self.generate_matrix(matrix, fake_label)
        # 将adj 中大于0的换为1， 小于等于0的不变
        A = torch.where(A > 0, torch.tensor(1.0).to(self.device), A)
        A = torch.mul(A, A_hat)
        # 构造复数adj
        adj = torch.cos(A) + 1j * torch.sin(A)
        # 计算message passing后的结果
        message = torch.matmul(adj, norm_features)
        new_fea = message
        # for ctensor in torch.flatten(new_fea):
        #     real = ctensor.real
        #     imag = ctensor.imag
        #     writer.add_histogram(f"l{self.name}/real", real.item(), epoch)
        #     writer.add_histogram(f"l{self.name}/imag", imag.item(), epoch)
        modulus = torch.abs(new_fea).to(self.device) + 1e-5
        # print("This is theta: ", self.theta)
        # print("This is matrix:", matrix)
        return new_fea / modulus


def complex_dist(distance_1, distance_2, p=1):
    real_part1 = distance_1.real
    imag_part1 = distance_1.imag
    real_part2 = distance_2.real
    imag_part2 = distance_2.imag

    real_distance = torch.cdist(real_part1, real_part2, p=p)
    imag_distance = torch.cdist(imag_part1, imag_part2, p=p)
    complex_distance = torch.sqrt(real_distance ** 2 + imag_distance ** 2)

    return complex_distance


def Augular_loss(theta, y_true,  m=0.2, s=0.2):
    '''
    m: maybe try 0.5 first
    s: 1.0  if over-fit, reduce; vice-versa
    '''
    # re place 0 => 1 and 1 => m in y_true
    num_classes = torch.max(y_true) + 1

# 将标签向量转换为 one-hot 形式的矩阵
    y_true = torch.nn.functional.one_hot(y_true, num_classes)
    # print(y_true.shape)
    M = (m - 1) * y_true + 1
    #print(M.shape)
    #print(theta.shape)
    # add appropriate margin to theta
    # M = torch.reshape(M, (8344,))
    new_theta = theta + M
    #print(new_theta)
    new_cos_theta = torch.cos(new_theta)

    # re-scale the cosines by a hyper-parameter s
    y_pred = s * new_cos_theta

    # the following part is the same as softmax loss
    numerators = torch.sum(y_true * torch.exp(y_pred), dim=1)
    denominators = torch.sum(torch.exp(y_pred), dim=1)
    loss = -torch.sum(torch.log(numerators / denominators))

    return loss


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, name):
        super(ComplexLinear, self).__init__()
        self.theta = nn.Parameter(torch.rand(in_features, out_features) * 2 * torch.pi)
        # self.name = name

    def forward(self, input, writer=None, epoch=0):
        mt = torch.cos(self.theta) + 1j * torch.sin(self.theta)
        fea = torch.matmul(input, mt)
        # for ctensor in torch.flatten(fea):
        #     real = ctensor.real
        #     imag = ctensor.imag
        #
        #     writer.add_histogram(f"fc{self.name}/real", real.item(), epoch)
        #     writer.add_histogram(f"fc{self.name}/imag", imag.item(), epoch)
        return fea


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def complex_dropout(self, input, p=0.5, training=True):
        device = input.device
        mask = torch.ones(input.real.shape, dtype=torch.float32, device=device)
        mask = F.dropout(mask, p, training) * 1 / (1 - p)
        mask.type(input.real.dtype)
        real, imag = mask * input.real, mask * input.imag
        return torch.complex(real, imag)

    def forward(self, input):
        if self.training:
            return self.complex_dropout(input, self.p)
        else:
            return input



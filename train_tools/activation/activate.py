# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2023/12/25 19:58
# @Author  : changqingai
# @FileName: activate.py
# ----------------------------

import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def draw_single_plot(x, y, x_name='x', y_name='y', img_path=''):
    plt.figure(figsize=(5, 2.5))
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if img_path:
        plt.savefig(img_path)
    plt.grid()
    plt.show()


def draw_multi_plot(value_list, x_name, y_name, title, img_path):
    """
    :param value_list: [x, y, name]
    :return:
    """
    fig, ax = plt.subplots()  # 创建图实例
    for x, y, name in value_list:
        ax.plot(x, y, label=name)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend()
    plt.grid()

    # 是否保存图片
    if img_path:
        plt.savefig(img_path)
        print("成功保存图片")
    plt.show()
    print("success")


def get_multi_activate_value():
    activate_list = []
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)

    # relu
    y = F.relu(x_)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'relu'])

    # sigmoid
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    y = F.sigmoid(x_)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'sigmoid'])

    # tanh
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    y = F.tanh(x_)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'tanh'])

    # swish
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    beta = 1
    y = x_ * F.sigmoid(x_ * beta)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'swish'])

    # silu
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    beta = 1
    threshold = 20
    y = F.silu(x_)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'swish'])

    # mish
    # x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    # y = x_ * F.tanh(F.softplus(x_, beta, threshold))
    # y.sum().backward()
    # activate_list.append([y, x_.grad, 'mish'])

    # gelu
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    y = F.gelu(x_)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'gelu'])

    # celu
    # x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    # y = F.celu(x_)
    # y.sum().backward()
    # activate_list.append([y, x_.grad, 'celu'])

    # elu
    x_ = torch.arange(-8.0, 8.0, 0.01, requires_grad=True)
    y = F.elu(x_)
    y.sum().backward()
    activate_list.append([y, x_.grad, 'elu'])
    return x_, activate_list


if __name__ == "__main__":

    x_, activate_list = get_multi_activate_value()
    act_value_list = [[x_.data.numpy(), obj[0].data.numpy(), obj[2]] for obj in activate_list]
    grad_value_list = [[x_.data.numpy(), obj[1].data.numpy(), obj[2]] for obj in activate_list]
    draw_multi_plot(act_value_list, x_name='x', y_name="激活值", title="激活函数对比", img_path='../../imgs/activation/act_multi.png')
    draw_multi_plot(grad_value_list, x_name='x', y_name="梯度值", title="激活函数梯度对比", img_path='../../imgs/activation/grad_multi.png')

    # 单个激活函数绘图
    draw_single_plot(x_.data.numpy(), activate_list[0][0].data.numpy(), x_name='x', y_name='y', img_path='')


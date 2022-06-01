"""This module is used for plotting, e.g, the loss and metrics curve from
Keras."""

import numpy as np
import plotly.graph_objects as go
from plotly import offline


def scatter_plotly_keras(history_dict, title=None, filename=None,
                         plot_loss=True):
    """使用 Keras 模型输出的训练结果字典，画出损失值和准确度的折线图。
    """

    train_loss = history_dict['loss']
    val_loss = history_dict.get('val_loss')
    train_accuracy = history_dict.get('accuracy')
    val_accuracy = history_dict.get('val_accuracy')
    epochs = list(range(1, len(train_loss) + 1))

    # 对于 YOLO-v4-CSP 模型，还有 AP 指标。
    p3_average_precision = history_dict.get('p3_AP')

    # 如果有准确度，则画出准确度曲线。如果没有准确度，则只需要画出损失值曲线。
    if plot_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, name='train_loss',
                                      mode='lines+markers'))
        if val_loss is not None:
            fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, name='val_loss',
                                          mode='lines+markers'))
        fig_loss.update_layout(title=f'{title}    \tloss_epoch',
                               xaxis_title='epochs', yaxis_title='loss')

        offline.plot(fig_loss, filename=f'{filename}_loss.html')

    if train_accuracy is not None:
        fig_object_exist_accuracy = go.Figure()
        fig_object_exist_accuracy.add_trace(
            go.Scatter(x=epochs, y=train_accuracy,
                       name='train_accuracy', mode='lines+markers'))
        fig_object_exist_accuracy.add_trace(
            go.Scatter(x=epochs, y=val_accuracy,
                       name='val_accuracy', mode='lines+markers'))
        fig_object_exist_accuracy.update_layout(
            title=f'{title}    \taccuracy_epoch',
            xaxis_title='epochs', yaxis_title='accuracy')

        offline.plot(fig_object_exist_accuracy,
                     filename=f'{filename}_accuracy.html')

    if p3_average_precision is not None:

        fig_object_exist_accuracy = go.Figure()

        fig_object_exist_accuracy.add_trace(go.Scatter(
            x=epochs, y=p3_average_precision,
            name='train_AP', mode='lines+markers'))

        fig_object_exist_accuracy.update_layout(
            title=f'{title}    \tAP_epoch',
            xaxis_title='epochs', yaxis_title='AP')

        offline.plot(fig_object_exist_accuracy,
                     filename=f'{filename}_AP.html')


def histogram_plotly(data_histogram, axis_names=None, histogram_title=None,
                     filename=None):
    """画一个直方图。

    输入：
        data_histogram：一个列表。列表里有1个元祖。
            元祖内有2个元素，分别代表x和y。x和y都是1D张量或列表。
        axis_names：一个列表，包含2个元素。2个元素是2个字符串，分别是x轴和y轴的名称。
        histogram_title：整个直方图的名称。
        filename：直方图文件的名称，必须包含后缀html，格式为'xxx.html'。如果保持为None，
        则只用于显示，不会在硬盘中保存该文件。

    输出：
        根据输入，用plotly画出的一个直方图。
    """

    if axis_names is None:
        raise ValueError('Please provide 2 names for the histogram.')

    x_axis_config = {'title': axis_names[0]}
    y_axis_config = {'title': axis_names[1]}

    x = data_histogram[0]
    y = data_histogram[1]
    data = [go.Bar(x=x, y=y)]
    histogram_layout = go.Layout(title=histogram_title,
                                 xaxis=x_axis_config,
                                 yaxis=y_axis_config)

    fig = {'data': data, 'layout': histogram_layout}

    offline.plot(fig, filename=f'{filename}.html')


def scatters_plotly(scatters_inputs, titles, file_name):
    """用 plotly 画一个折线图，图中可以有多条折线。

    Arguments:
        scatters_inputs: 是一个列表，形式是 [(x1, y1, trace_name1), (x2, y2,
            trace_name2), ...]，列表内的每个元祖代表一个折线图。
            x 是一个序列（比如列表等），其中为若干浮点数，代表输入的横坐标值。
            y 是一个序列（比如列表等），其中为若干浮点数，代表纵坐标值，并且数量必须和 x
            的数量相同。
            trace_name，是一个字符串，字符串代表一条折线的名字。
        titles: 一个元祖，包含 3 个字符串，第一个字符串是整个折线图的名字，第二个字符串
            是横坐标名字，第三个字符串是纵坐标名字。
        file_name: 一个字符串，是这个折线图文件的名字，格式是 'xxx.html' 。
    Returns:
        返回一个 html 格式的折线图文件，自动保存在当前文件夹中。
    """

    fig = go.Figure()
    for each_scatter in scatters_inputs:
        x = each_scatter[0]
        y = each_scatter[1]
        trace_name = each_scatter[2]

        fig.add_trace(go.Scatter(x=x, y=y, name=trace_name,
                                 mode='lines+markers'))

    title = titles[0]
    xaxis_title = titles[1]
    yaxis_title = titles[2]
    fig.update_layout(title=title,
                      xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    offline.plot(fig, filename=file_name)

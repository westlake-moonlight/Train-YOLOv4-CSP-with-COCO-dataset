"""该模块专门用于创建 tf.data.dataset"""

import copy
import json
import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 导入自定义的全局变量和函数
from yolo_v4_csp import FEATURE_MAP_P5, FEATURE_MAP_P4, FEATURE_MAP_P3
from yolo_v4_csp import ANCHOR_BOXES_P5, ANCHOR_BOXES_P4, ANCHOR_BOXES_P3
from yolo_v4_csp import MODEL_IMAGE_SIZE
from yolo_v4_csp import ciou_calculator

MODEL_IMAGE_HEIGHT = MODEL_IMAGE_SIZE[0]
MODEL_IMAGE_WIDTH = MODEL_IMAGE_SIZE[1]

NUM_CLASSES = 80  # 使用 COCO 中的 80 个类别

# 此部分为模块常量 module constants，所以名字用大写字母，模块常量可以用于各个函数中 ===
# CATEGORY_NAMES_TO_DETECT 是所有需要探测的类别的名字。
CATEGORY_NAMES_TO_DETECT = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

# COCO2017 有 80 个类别，但是因为部分 id 空缺，所以最大 id 编号为 90.
# COCO2017 中，最多标注数量的一张图片，标注了 93 个物体。但是标注数量的中位数，是 4 个标注。

# 以下为模块常量 module constants. 默认 TRAIN_ANNOTATIONS_DICT 为训练集的标注文件，
# 后续使用时，可以直接使用该常量。需要用函数 _get_annotations_dict_coco 进行生成。
TRAIN_ANNOTATIONS_DICT = {}
# TRAIN_ANNOTATIONS_RAW_DICT： 一个字典，是由 COCO 的标注文件转化而来。包含的5个键 key
# 为：'info', 'licenses', 'images', 'annotations', 'categories' 。
TRAIN_ANNOTATIONS_RAW_DICT = {}


def _get_annotations_dict_coco(dataset_type, bbox_area_ascending=True):
    """把标注文件的信息转换为一个字典，并将该字典返回。字典的各个 key，就是图片编号，而 key
    对应的值 value，则是一个列表，该列表包含了此图片的所有标注。标注文件为 COCO 格式。

    Arguments:
        dataset_type： 一个字符串，指定是训练集 'train' 或者验证集 'validation'。
        bbox_area_ascending: 一个布尔值。如果为 True，则把每个图片的物体框，按照面积从小
            到大的进行排序。如果为 False，则不进行任何排序。

    Returns:
        annotations_dict： 一个字典，包含了 COCO 标注文件中，用于探测任务的所有标注信息。
            该字典的形式为：{'139':annotations_list_139,
            '285':annotations_list_285, ...}
            对于每一个有标注的图片，都会在字典 annotations_dict 中生成一个键值对。键
            key 是图片的编号，如上面的 '139'，值 value 则是一个标注列表，包含了该图片
            所有的标注。如上面的 annotations_list_139。
            每一个标注列表中，可以有若干个标注，并且每个标注也是一个列表，它的形式是：
            annotations_list_139 = [[annotation_1], [annotation_2], ...] 。所以，
            如果在 annotations_list_139 中包含20个列表，也就意味着图片 '139' 有 20 个
            标注。
            对于每一个标注，它的形式是 annotation_1=[category_id, center_point_x,
                center_point_y, height, width, bbox_area]
            其中，bbox_area 是探测框的面积。所以每个标注共有 6 个元素，标注的中间 4 位是
            探测框的中心点坐标，以及高度宽度。

        annotations_raw_dict: 一个字典，是从 COCO 的标注文件直接转化而来的原始字典。包含
            5个键：'info', 'licenses', 'images', 'annotations', 'categories'。

    """

    # 设置好 instances_train2017.json 和 instances_val2017.json 的路径。
    train_annotations = (
        r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
        r'\annotations_trainval2017\annotations\instances_train2017.json')
    validation_annotations = (
        r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
        r'\annotations_trainval2017\annotations\instances_val2017.json')

    if dataset_type == 'train':
        path_annotations = train_annotations
    elif dataset_type == 'validation':
        path_annotations = validation_annotations
    else:
        path_annotations = None

    try:
        with open(path_annotations) as f:
            annotations_raw_dict = json.load(f)
    except FileNotFoundError:
        print(f'File not found: {path_annotations}')

    annotations_dict = {}
    # =======================做一个进度条。============================
    progress_bar = keras.utils.Progbar(
        len(annotations_raw_dict['annotations']), width=60, verbose=1,
        interval=0.5, stateful_metrics=None, unit_name='step')
    print(f'Extracting the annotations for {dataset_type} dataset ...')
    # =======================做一个进度条。============================

    # annotations_raw_dict['annotations'] 是一个列表，存放了所有的标注。
    # 而每个标注本身，则是一个字典，包含的key为：(['segmentation', 'area', 'iscrowd',
    # 'image_id', 'bbox', 'category_id', 'id'])
    fixed_records = []
    for i, each_annotation in enumerate(annotations_raw_dict['annotations']):
        progress_bar.update(i)

        # image_id 本身是一个整数，需要把它转换为字符，才能作为字典的key使用。
        image_id = str(each_annotation['image_id'])
        category_id = each_annotation['category_id']
        # bbox 是一个列表。
        bbox = each_annotation['bbox']
        top_left_x = bbox[0]
        top_left_y = bbox[1]
        # 注意标注文件中，bbox[2] 和 bbox[3] 是 bbox 的宽度和高度。
        width = bbox[2]
        height = bbox[3]

        center_point_x = top_left_x + width / 2
        center_point_x = round(center_point_x, 3)
        center_point_y = top_left_y + height / 2
        center_point_y = round(center_point_y, 3)

        # 有几张图片的高度或宽度为 0 ，直接将其设置为 1，以免遗漏该目标。（例如图片编号为
        # 200365 的图片，有根香肠的高度被标为了 0，需要将其改为 1.）
        if np.isclose(width, 0):
            width = 1
            one_record = ['Width', i, image_id, category_id,
                          center_point_x, center_point_y]
            fixed_records.append(one_record)
        elif np.isclose(height, 0):
            height = 1
            one_record = ['Height', i, image_id, category_id,
                          center_point_x, center_point_y]
            fixed_records.append(one_record)

        bbox_area = round(width * height, 1)

        # 需要把标注信息和图片对应起来，放到字典 annotations_dict 中。
        if image_id not in annotations_dict:
            # 如果还未记录该图片的任何标注信息，则先要把该图片建立为一个key。
            annotations_dict[image_id] = []
            first_annotation = [category_id, center_point_x, center_point_y,
                                height, width, bbox_area]
            annotations_dict[image_id].append(first_annotation)
        else:
            later_annotation = [category_id, center_point_x, center_point_y,
                                height, width, bbox_area]
            annotations_dict[image_id].append(later_annotation)

        # 这一部分检查是否有坐标值为负数的情况。
        if (bbox[1] < 0) or (bbox[0] < 0):
            print(f'Bbox error! Annotation index: {i}, image_id: {image_id}, '
                  f'category_id: {category_id}.\nIn "annotations" section: '
                  f'bbox coordinates are smaller than 0.\n'
                  f'bbox[0]: {bbox[0]}, bbox[1]: {bbox[1]}\n')

    if bbox_area_ascending:
        # 把各个物体框，按照面积由小到大的顺序进行一下排序。
        for key, annotations in annotations_dict.items():
            annotations_dict[key] = sorted(
                annotations, key=lambda annotation: annotation[-1])

    # 如果有错误信息，则输出所有错误信息。
    if len(fixed_records) > 0:
        print(f'\nDone. Here are {len(fixed_records)} fixed records.')
        for one_record in fixed_records:
            print(f'{one_record[0]} was 0, set to 1. '
                  f'\tImage: {one_record[2]}\tcategory_id: {one_record[3]},\t'
                  f'annotation index: {one_record[1]},\t'
                  f'object center {one_record[4]:.1f}, {one_record[5]:.1f}, ')

    return annotations_dict, annotations_raw_dict


# noinspection PyRedeclaration
TRAIN_ANNOTATIONS_DICT, TRAIN_ANNOTATIONS_RAW_DICT = _get_annotations_dict_coco(
    dataset_type='train')

VALIDATION_ANNOTATIONS_DICT = None
# noinspection PyRedeclaration
VALIDATION_ANNOTATIONS_DICT, _ = _get_annotations_dict_coco(
    dataset_type='validation')


# 以下为 2 个模块常量 module constants. 用函数 _coco_categories_to_detect 进行生成。
CATEGORIES_TO_DETECT = 0
FULL_CATEGORIES = 0


def _coco_categories_to_detect():
    """根据所要探测的类别名字，将类别名字、类别 id，类别所属的大类这 3 者关联起来，存储到
    一个 Pandas 的 DataFrame 中，并返回该 DataFrame。

    Returns:
        categories_to_detect: 一个 Pandas 的 DataFrame，包含了所有要探测的类别。
            表格里包括了一一对应的 4 类信息：id_in_model，id_in_coco，类别名字，以及该
            类别所属的大类。id_in_coco 是 COCO 中的 id 编号，id_in_model 是转换到模型
            中的编号，虽然 COCO 中只有 80 个类别，但是最大的 id 编号为 90，即最大的
            id_in_coco 为 90，而转换到模型中之后，最大的编号依然为 80，即最大的
            id_in_model 为 80。
        full_categories： 一个 Pandas 的 DataFrame。包含了 COCO 标注文件里所有 80 个
            类别，并且 id，类别名字和该类别所属的大类一一对应。

    """

    full_categories = pd.DataFrame({})
    for i, each in enumerate(TRAIN_ANNOTATIONS_RAW_DICT['categories']):
        id_in_model = i
        full_categories.loc[id_in_model, 'id_in_model'] = id_in_model
        full_categories.loc[id_in_model, 'id_in_coco'] = each['id']
        full_categories.loc[id_in_model, 'name'] = each['name']
        full_categories.loc[id_in_model, 'supercategory'] = each[
            'supercategory']

    categories = full_categories[
        full_categories['name'].isin(CATEGORY_NAMES_TO_DETECT)]

    categories_to_detect = categories.set_index('id_in_model')

    return categories_to_detect, full_categories


# noinspection PyRedeclaration
CATEGORIES_TO_DETECT, FULL_CATEGORIES = _coco_categories_to_detect()


def _get_object_boxes_coco(one_image_path, image_original_size,
                           annotations_dict=None):
    """根据一个图片的完整路径，提取该图片中的所有物体框并返回。

    Arguments:
        one_image_path： 一个字符串，是一张图片的完整路径。
        image_original_size: 是一个元祖，由 2 个元素组成，表示图片的初始大小，形式为
            (height, width)。
        annotations_dict: 是一个字典，包含了图片的所有标注文件。如果是训练集图片的标注
            文件，则不需要输入，直接使用模块常量 TRAIN_ANNOTATIONS_DICT。

    Returns:
        object_boxes: 一个列表，包括了一个图片中所有的物体框。列表的每一个元素代表一个
            物体框。而每一个物体框，都是一个长度为 85 的元祖。元祖的第 0 位，代表一个
            probability，即该物体框是否有物体的概率。
            元祖的第 1 位到第 81 位，是 one-hot 编码，代表 COCO 数据集的 80 个类别。
            元祖的最后 4 位，是物体框的信息。按顺序分别是中心点横坐标和纵坐标，物体框的高
            度和宽度，格式为： (center_x, center_y, height, width)
            数据类型为 int 加 float32，输出以后会全部被转换为 float32。

    """
    # 如果 annotations_dict 为 None，则使用提前生成的训练集 TRAIN_ANNOTATIONS_DICT。
    if annotations_dict is None:
        annotations_dict = TRAIN_ANNOTATIONS_DICT

    # 因为是在 tf.py_function 下，即 eager 模式，所以下面可以使用 numpy()
    # COCO 2017 最大的图片名字是 000000581929.jpg
    image_name = str(one_image_path.numpy())[-15: -5]

    # 还要转换为整数，去掉 image_name 前面的 0，然后再转换回字符串类型
    image_name = int(image_name)
    image_name = str(image_name)

    # 这一部分先计算图片缩放之后产生的黑边 blank_in_height和 blank_in_width，后续
    # 计算探测框的坐标时会用到。
    image_original_height, image_original_width = image_original_size

    width_scale = image_original_width / MODEL_IMAGE_WIDTH
    height_scale = image_original_height / MODEL_IMAGE_HEIGHT
    resize_scale = None
    blank_in_height = 0
    blank_in_width = 0
    if width_scale > height_scale:
        resize_scale = width_scale
        resized_height = image_original_height / resize_scale
        blank_in_height = (MODEL_IMAGE_HEIGHT - resized_height) / 2
    elif width_scale == height_scale:
        resize_scale = width_scale
    elif width_scale < height_scale:
        resize_scale = height_scale
        resized_width = image_original_width / resize_scale
        blank_in_width = (MODEL_IMAGE_WIDTH - resized_width) / 2

    # image_annotations 是一个列表，包含了该图片的所有标注。并且是一个标志。
    # 因为有些图片没有任何标注，所以要用get方法，此时标志 image_annotations 将为空。
    image_annotations = annotations_dict.get(image_name)

    # 注意，因为后续的代码会把 image_annotations 清空，所以此处应该使用拷贝，
    # 重新生成一个列表。而且因为列表属于mutable，还必须使用 deepcopy，否则原字典
    # annotations_dict 也会被清空。
    image_annotations = copy.deepcopy(image_annotations)

    # labels 用于存放一个图片的所有标注，其中每一个标注都是一个长度为 85 的元祖
    object_boxes = []

    while image_annotations:
        # 每一个标注是一个列表，含有 6 个元素，分别代表 (category_id_in_coco,
        # center_point_x, center_point_x, bbox_height, bbox_width, bbox_area)
        one_annotation = image_annotations.pop(0)

        id_in_coco = one_annotation[0]
        # 以下if语句，用于判断 id_in_coco 是否在表格 CATEGORIES_TO_DETECT 的
        # 'id_in_coco' 这一列中。
        if (CATEGORIES_TO_DETECT['id_in_coco'].isin([id_in_coco])).any():
            # 根据表格 CATEGORIES_TO_DETECT，把 id_in_coco 转换为 id_in_model。
            category = CATEGORIES_TO_DETECT[
                CATEGORIES_TO_DETECT['id_in_coco'] == id_in_coco]
            # id_in_model 是 0 到 79 的整数，代表 80 个类别。
            id_in_model = int(category.index[0])  # 从 np.float64 转换为 int
            # 用 to_categorical 转换成 one-hot 编码，如果一共有 3 个类别，用 1,0,0 表
            # 示第 0 类， 0,1,0 表示第 1 类， 0,0,1 表示第 2 类
            categorical_id = keras.utils.to_categorical(
                id_in_model, num_classes=NUM_CLASSES)

            center_point_x = one_annotation[1]
            center_point_y = one_annotation[2]
            bbox_height = one_annotation[3]
            bbox_width = one_annotation[4]

            # 以下将坐标点和高宽转换为 608x608 大小图片中的实际值。
            center_point_x = center_point_x / resize_scale
            center_point_y = center_point_y / resize_scale
            bbox_height = bbox_height / resize_scale
            bbox_width = bbox_width / resize_scale

            # 原图在缩放到 608x608 大小并且居中之后，物体框中心点会发生移动。
            if width_scale >= height_scale:
                center_point_y += blank_in_height

            elif width_scale < height_scale:
                center_point_x += blank_in_width

            # 把 bbox 的 4 个参数，从 tf 张量转换为数值。
            center_point_x = center_point_x.numpy()
            center_point_y = center_point_y.numpy()
            bbox_height = bbox_height.numpy()
            bbox_width = bbox_width.numpy()
            # one_object_box 是一个元祖，长度为 85 。第 0 位设置为 1， 表示该物体
            # 框中有物体；第 1 到第 80 位是类别的 one-hot 编码。最后 4 位是物体框信息。
            one_object_box = (1, *categorical_id, center_point_x,
                              center_point_y, bbox_height, bbox_width)

            object_boxes.append(one_object_box)

    return object_boxes


def _get_paths_image_coco(path_image, images_range=None, shuffle_images=False):
    """从文件夹中提取图片，返回图片的名字列表。

    Arguments:
        path_image： 一个字符串，是所有图片的存放路径。
        images_range: 一个元祖，是图片的索引范围，如(0, 5000)。
        shuffle_images： 一个布尔值，如果为 True，则把全部图片的顺序打乱。

    Returns:
        paths_image: 一个列表，该列表是所有被选中图片的完整路径，例如：
            ['D:\\COCO_datasets\\COCO_2017\\train2017\\000000000009.jpg', ...]。
            列表的大小，由 images_range 设定。如不设定 images_range，将默认使用
            文件夹内全部图片。
    """

    paths_image = []
    # os.walk 会进入到逐个子文件中，找出所有的文件。
    for path, dir_names, image_names in os.walk(path_image):
        for image_name in image_names:
            one_image_path = os.path.join(path, image_name)
            paths_image.append(one_image_path)

    if shuffle_images:
        random.shuffle(paths_image)

    # 如果指定了图片范围，则只使用一部分的图片。
    if images_range is not None:
        start_index, end_index = images_range
        paths_image = paths_image[start_index: end_index]

    return paths_image


def _get_image_tensor_coco(one_image_path):
    """根据输入的图片路径，转换为一个 3D TF 张量并返回。

    Arguments:
        one_image_path： 一个字符串，是一张图片的完整路径。
    
    Returns:
        image_tensors: 一个图片的 3D 张量，形状为 (height, width, 3)。
        image_original_size: 是一个元祖，由2个元素组成，表示图片的初始大小，形式为
            (height, width)。后续根据标注文件生成标签时要用到这个列表里的信息。
    """

    image_file = tf.io.read_file(one_image_path)
    original_image_tensor = tf.image.decode_image(image_file, channels=3)

    # image_original_size 必须分开作为两个数处理，不能直接用shape[: 2]，否则会得到
    # 一个 shape 张量，而不是一个元祖
    image_original_size = (original_image_tensor.shape[0],
                           original_image_tensor.shape[1])

    image_tensor = tf.image.resize_with_pad(
        original_image_tensor, target_height=MODEL_IMAGE_HEIGHT,
        target_width=MODEL_IMAGE_WIDTH)

    # 将图片数值限制在 [0, 255] 范围。遥感图像的像素有负数。
    image_tensor = tf.clip_by_value(
        image_tensor, clip_value_min=0, clip_value_max=255)

    image_tensor /= 127.5  # 将图片转换为 [0, 2] 范围
    image_tensor -= 1  # 把图片转换为 [-1, 1] 范围。

    return image_tensor, image_original_size


def _get_label_arrays(one_image_path, image_original_size, dataset_type):
    """根据输入的图片路径，创建其对应的标签数组 p5_label, p4_label, p3_label。

    5 个步骤如下:
    1. 创建 3 个 Numpy zeros 数组 p5_label, p4_label, p3_label，形状分别为
        (*FEATURE_MAP_P5, 3, 85), (*FEATURE_MAP_P4, 3, 85), (*FEATURE_MAP_P3,
        3, 85)。把 3 个特征层的预设框参数，分别填入对应的这 3 个数组中。
    2. 将特征图每个方框 cell 的中心点坐标，以及预设框高度 ph 和宽度 pw，填入数组
        p5_label, p4_label, p3_label 中。
        每个特征图位置上，都有不同长宽比例的 3 个预设框，按照全局变量 ANCHOR_BOXES
        中的顺序，把预设框填入长度为 85 的向量，向量的最后 4 位依次是(x, y, ph, pw)。
        而填入预设框信息的原因，是为了后面计算预设框和物体框的 IOU。
    3. 取得该图片中的所有物体框。
    4. 遍历该图片中的所有物体框，对每一个物体框，将其信息填入标签 p5_label, p4_label,
        p3_label 中的某一个位置。
        4.1 对当前物体框，创建 3 个和 p5_label, p4_label, p3_label 形状相同的数组
            object_array。
        4.2 找出 label 中还没有被其它物体框占用的位置，用这些位置来计算它们和当前物体框
            的 CIOU。
        4.3 把当前物体框的类别和大小等信息，填入 3 个数组 object_array 中。
           即 85 位长度的向量的 第 0 位填为 1， 中间 80 位填写该物体的分类，最后 4 位填写
           该物体框的参数，格式为 (x, y, height, width)。这里的 x, y, height, width
           4 个参数，都是一个在 608x608 大小图片中的实际值，而不是比例值。
        4.4 计算当前物体框和预设框之间的 CIOU。
           这里之所以不用 for 循环遍历预设框的方法计算重叠面积，是因为用数组计算通常比
           循环遍历的速度更快。
        4.5 找出最大的 CIOU 所对应的预设框位置。
        4.6 将当前物体框信息填入到最大 CIOU 的预设框位置。
        4.7 重复上面步骤 4.1 到 4.6，直到遍历完图片内所有的物体框。
    5. 最终返回标签数组 p5_label, p4_label, p3_label，和物体框重叠最大的预设框，记录了
        物体框的完整信息。那些没有物体的预设框，其长度为 85 向量的前 81 位，将都是 0。

    Arguments:
        one_image_path: 一个字符串，图片存放路径。
        image_original_size: 是一个元祖，由2个元素组成，表示图片的初始大小，形式为
            (height, width)。
        dataset_type: 一个字符串，指定是训练集 'train' 或者验证集 'validation'。

    Returns:
        p5_label: 是一个 float32 型张量，形状为 (*FEATURE_MAP_P5, 3, 85)。
        p4_label: 是一个 float32 型张量，形状为 (*FEATURE_MAP_P4, 3, 85)。
        p3_label: 是一个 float32 型张量，形状为 (*FEATURE_MAP_P3, 3, 85)。
    """

    # 步骤 1，创建 p5_label, p4_label, p3_label。
    p5_label = np.zeros(shape=(*FEATURE_MAP_P5, 3, 85), dtype=np.float32)
    p4_label = np.zeros(shape=(*FEATURE_MAP_P4, 3, 85), dtype=np.float32)
    p3_label = np.zeros(shape=(*FEATURE_MAP_P3, 3, 85), dtype=np.float32)

    # 步骤 2.1，将方框中心点坐标 cell_center 填入对应的数组中。
    # 注意一点，根据 YOLO V3 论文 2.1 节第一段以及配图 figure 2，cx_cy 是每一个 cell
    # 的左上角点，这样预测框的中心点 bx_by 才能达到该 cell 中的每一个位置。
    feature_maps = FEATURE_MAP_P5, FEATURE_MAP_P4, FEATURE_MAP_P3
    px_labels = p5_label, p4_label, p3_label
    for feature_map_px, px_label in zip(feature_maps, px_labels):
        # 以 p5 为例，构造一个 19 x 19 大小的网格
        grid = np.ones(shape=feature_map_px, dtype=np.int32)
        cx_cy = np.argwhere(grid)  # argwhere 函数可以获取非 0 元素的索引值

        # cx_cy 的形状为 (361, 2)， 361 = 19 x 19，下面将其形状变为 (19, 19, 1, 2)
        cx_cy = cx_cy.reshape(*feature_map_px, 1, 2)
        # 需要把预设框放在每个网格的中心点，所以要加上 0.5，例如(12, 13)变成 (12.5, 13.5)
        # 其实如果不移到中心点位置，计算物体框和预设框的 IOU 时，结果应该也是一样的
        cell_center = cx_cy + 0.5

        # 必须将 cell_center 转换为 608x608 大小图片中的实际值，才能用于计算 IOU。
        scale_height = MODEL_IMAGE_SIZE[0] / feature_map_px[0]
        scale_width = MODEL_IMAGE_SIZE[1] / feature_map_px[1]
        scale_in_image = scale_height, scale_width
        cell_center *= scale_in_image

        # p5_label 的形状为(19, 19, 3, 85)，在最后一个维度的 85 位数中，其倒数第 4 位
        # 和倒数第 3 位，是 cell 的中心点坐标 cell_center
        px_label[..., -4: -2] = cell_center

    # 步骤 2.2，将 ANCHOR_BOXES 中的预设框高度 ph 和宽度 pw 填入对应的数组中。
    for i in range(3):
        # 填入预设框的 height, width 信息
        p5_label[:, :, i, -2:] = ANCHOR_BOXES_P5[i]
        p4_label[:, :, i, -2:] = ANCHOR_BOXES_P4[i]
        p3_label[:, :, i, -2:] = ANCHOR_BOXES_P3[i]

    # 步骤 3，取得该图片中的所有物体框 object_boxes，object_boxes 是一个列表，列表中
    # 的每一个元素都是一个长度为 85 的元祖，该元祖代表一个物体框。

    # 使用已经提前生成的 ANNOTATIONS_DICT。
    if dataset_type == 'train':
        annotations_dict = TRAIN_ANNOTATIONS_DICT
    else:
        annotations_dict = VALIDATION_ANNOTATIONS_DICT

    object_boxes = _get_object_boxes_coco(
        one_image_path=one_image_path, image_original_size=image_original_size,
        annotations_dict=annotations_dict)

    # 步骤 4，将每个物体框填入数组 p5_label, p4_label, p3_label 中的某一个位置。
    for object_box in object_boxes:
        # 步骤 4.1，创建 3 个和 p5_label, p4_label, p3_label 形状相同的 object_array
        # object_array_p5 形状为 (19, 19, 3, 85)。
        object_array_p5 = np.ones(shape=(*FEATURE_MAP_P5, 3, 85),
                                  dtype=np.float32)  # 设定为float32，默认float64
        # 下面的 p4，p3 进行相同操作。
        object_array_p4 = np.ones(shape=(*FEATURE_MAP_P4, 3, 85),
                                  dtype=np.float32)
        object_array_p3 = np.ones(shape=(*FEATURE_MAP_P3, 3, 85),
                                  dtype=np.float32)

        # 步骤 4.2，找出 label 中还没有被其它物体框占用的位置，用这些位置来计算它们和
        # 当前物体框的 IOU。
        # occupied_p5 是一个布尔张量，形状为 (19, 19, 3)，用于找出所有已经填入物体框
        # 的位置。有物体框的位置，其 85 位长度向量的第 0 位是 1.
        # 不可以直接比较浮点数是否相等，应该用 isclose 函数
        occupied_p5 = np.isclose(p5_label[..., 0], 1)
        occupied_p4 = np.isclose(p4_label[..., 0], 1)
        occupied_p3 = np.isclose(p3_label[..., 0], 1)

        # 步骤 4.3，对 object_array 中每一个 85 位长度的向量，都填入 object_box
        # 的信息。object_box 是一个长度为 85 的元祖，包含了当前物体框的类别和大小等信息。
        # 其中，该物体框的中心点位置和大小信息 x, y, height, width 这 4 个参数，都是
        # 在 608x608 大小图片中的实际值，而不是比例值。
        # 对于已经填入物体框信息的位置，就不再参与计算 IOU，所以索引方式为取反，即 ~ 符号，
        # 将把布尔数组中为 False 的位置变为 True，即在没有填入物体框的地方变为 True。
        object_array_p5[~occupied_p5] = object_box
        object_array_p4[~occupied_p4] = object_box
        object_array_p3[~occupied_p3] = object_box

        # 步骤 4.4 计算当前物体框和预设框之间的 CIOU。
        # 传统的 IOU 有一个问题，当一个较大的物体框覆盖多个网格时，可能会和多个预测框有相同
        # 的 IOU，但是如果用 CIOU，则不容易出现此问题，因为 CIOU 会选出距离最近的预测框。
        # 3 个 ciou 数组的形状分别为 (19, 19, 3),(38, 38, 3),(76, 76, 3)。
        ciou_loss_p5 = ciou_calculator(
            label_bbox=p5_label[..., -4:],
            prediction_bbox=object_array_p5[..., -4:])
        ciou_loss_p4 = ciou_calculator(
            label_bbox=p4_label[..., -4:],
            prediction_bbox=object_array_p4[..., -4:])
        ciou_loss_p3 = ciou_calculator(
            label_bbox=p3_label[..., -4:],
            prediction_bbox=object_array_p3[..., -4:])

        # 因为 ciou_calculator 计算的是损失值，所以最大的 CIOU 实际上是最小的损失值。
        min_ciou_loss_p5 = np.amin(ciou_loss_p5)
        min_ciou_loss_p4 = np.amin(ciou_loss_p4)
        min_ciou_loss_p3 = np.amin(ciou_loss_p3)
        min_ciou = np.amin((min_ciou_loss_p5,
                            min_ciou_loss_p4, min_ciou_loss_p3))

        # 步骤 4.5 找出最大的 CIOU 所对应的预设框位置。
        # 3 个布尔数组 min_ciou_mask_p5, min_ciou_mask_p4, min_ciou_mask_p3 的形
        # 状分别为 (19, 19, 3),(38, 38, 3),(76, 76, 3)，一般只有一个位置为 True
        min_ciou_mask_p5 = np.isclose(ciou_loss_p5, min_ciou)
        min_ciou_mask_p4 = np.isclose(ciou_loss_p4, min_ciou)
        min_ciou_mask_p3 = np.isclose(ciou_loss_p3, min_ciou)

        true_quantity = (np.sum(min_ciou_mask_p5) +
                         np.sum(min_ciou_mask_p4) + np.sum(min_ciou_mask_p3))
        # 使用 CIOU 时，存在最多 4 个物体框预测同一个物体的情况：即中心点在 4 个框的交点，
        # 且物体框完整覆盖 4 个单元格时。且这 4 个物体框一定处于同一个特征层。（不同特征层
        # 的预设框大小不同，所以 IoU 必然不同, CIOU 也不同。）
        if true_quantity > 1:
            # 如果有 2 个或 4 个物体框的 CIOU 都相同，则按 2 个条件进行选择：1. 选择离
            # 中心点的距离最远的物体框（因为中心位置通常物体框比较多，此举可以把物体框疏散
            # 一下）。2.如果有多个物体框，到中心点的最远距离相同（这些物体框关于中心点对称，
            # 或是关于中轴对称），则选择最先出现的物体框。（也就是行索引，列索引最小的物体
            # 框，并且是按照先搜索行索引，再搜索列索引的顺序）
            # 具体 5 个操作步骤如下：
            # 1. 取出具有 min_ciou 物体框的索引值。2. 计算 min_ciou 物体框到中心点的
            # 距离。3. 找出具有最大距离的物体框。4.如果具有最大距离的物体框，数量多于 1，
            # 则使用最先出现的 True。5. 安装 p3, p4, p5 的顺序进行搜索。如果在 p3 中
            # 找到一个合适的预设框，则跳出循环，不再搜索 p4, p5.（如果不这样做，编号为
            # 26812 的图片在 p4, p5 特征层，都会有一个预设框对应同一个物体）

            min_ciou_masks = [min_ciou_mask_p3,
                              min_ciou_mask_p4, min_ciou_mask_p5]

            # min_ciou_mask 形状为 (*FEATURE_MAP_Px, 3)
            for min_ciou_mask in min_ciou_masks:

                if np.sum(min_ciou_mask) > 0:
                    # 下面把 min_ciou_mask 的 True 数量修剪为 1 个

                    # 1. 取出具有 min_ciou 物体框的索引值 min_ciou_indexes。
                    # min_ciou_indexes 的形状为 (n, 3)
                    # 假如其中某个元素的值是 (5, 8, 1)，代表该物体框在特征图中的索引位置
                    # 是 5, 8, 1.
                    min_ciou_indexes = np.argwhere(min_ciou_mask)

                    # 2. 计算 min_ciou 物体框到中心点的距离。
                    # feature_map_center 是一个 float64 型张量，形状为 (2,)
                    feature_map = tf.shape(min_ciou_mask)[: 2]
                    feature_map_center = feature_map / 2

                    # 求出到中心点的边长 edges， edges 的形状为 (n, 2)。注意对
                    # min_ciou_indexes，只需要使用前 2 位参与计算距离值。
                    edges = min_ciou_indexes[:, : 2] - feature_map_center

                    # l2_distance 的形状为 (n)
                    l2_distances = tf.math.reduce_euclidean_norm(edges, axis=-1)

                    # l2_distance_max 是一个标量.
                    l2_distance_max = tf.math.reduce_max(l2_distances)

                    # 3. 找出具有最大距离的物体框。max_distance_mask 是一个布尔数组，
                    # 形状为 (n)，如果为 True 则表示某个物体框和中心点具有最大距离.
                    max_distance_mask = np.isclose(l2_distances,
                                                   l2_distance_max)

                    # 如果 max_distance_mask 只有 1 个 Ture，说明已经找到唯一的预设框。
                    if np.sum(max_distance_mask) == 1:
                        # bbox_position 是一个向量，记录了一个标签 bbox 的索引值，
                        # 形状为 (3,)
                        one_true_bbox_position = min_ciou_indexes[
                            max_distance_mask]

                    # 4.如果具有最大距离的物体框，数量多于 1，则使用最先出现的 True。
                    else:
                        # max_distance_bbox_indexes 是一个 2D 数组，形状为 (m, 1)，
                        # 表示有 m 个物体框，它们到特征图中心点的最大距离都相同。
                        max_distance_bbox_indexes = np.argwhere(
                            max_distance_mask)

                        # first_true_bbox 是一个整数型数组，形状为 (1,)，记录了在
                        # max_distance_mask 中，最先出现的 True。
                        first_true_bbox = max_distance_bbox_indexes[0]

                        # one_true_bbox_position 是一个数组，形状为 (1, 3).
                        # 记录了最先出现 True 的 bbox 在特征图中的位置。
                        one_true_bbox_position = min_ciou_indexes[
                            first_true_bbox]

                    # 需要生成一个新的 mask_array 数组。和 min_ciou_mask 形状
                    # 相同，都为 (*FEATURE_MAP_Px, 3)
                    mask_array = np.zeros_like(min_ciou_mask)

                    # 设置这个 mask_array 中只有一个 True。并且因为
                    # one_true_bbox_position 是一个 2D数组，这里必须取第 0 位。
                    mask_array[tuple(one_true_bbox_position[0])] = 1

                    # 将 mask_array 转换为布尔数组。
                    mask_array = (mask_array == 1)

                    # 把 mask_array 赋值给 3 个 min_ciou_mask_px 其中之一。
                    if min_ciou_mask_p5.shape == mask_array.shape:
                        min_ciou_mask_p5 = mask_array
                        # 找到一个合适的预设框之后，就应该把其它位置的预设框设为 False
                        min_ciou_mask_p4 *= False
                        min_ciou_mask_p3 *= False

                    elif min_ciou_mask_p4.shape == mask_array.shape:
                        min_ciou_mask_p4 = mask_array
                        min_ciou_mask_p5 *= False
                        min_ciou_mask_p3 *= False

                    elif min_ciou_mask_p3.shape == mask_array.shape:
                        min_ciou_mask_p3 = mask_array
                        min_ciou_mask_p5 *= False
                        min_ciou_mask_p4 *= False

                    # 已经找到一个合适的预设框之后，就无须搜索后面的 p4, p5 特征层，
                    # 所以用 break 跳出循环。
                    break

        # 再次检查 true bbox 的数量。
        true_quantity = (np.sum(min_ciou_mask_p5) +
                         np.sum(min_ciou_mask_p4) +
                         np.sum(min_ciou_mask_p3))

        if true_quantity > 1:
            print(f'\nWrong function! There are {true_quantity} true '
                  f'positions.\ttrue_positions_p5: {np.sum(min_ciou_mask_p5)}'
                  f', \ttrue_positions_p4:{np.sum(min_ciou_mask_p4)},\t'
                  f'true_positions_p3: {np.sum(min_ciou_mask_p3)}. '
                  f'\nImage: {one_image_path}\n')

        # 3 个 min_ciou_mask 经过修改，将只有一个 True，已经可以作为 mask 使用。
        mask_p5 = min_ciou_mask_p5
        mask_p4 = min_ciou_mask_p4
        mask_p3 = min_ciou_mask_p3

        # 步骤 4.6 将当前物体框信息填入到最大 CIOU 的预设框位置。
        # 物体框信息 object_box 将只会写入到标签 p5_label, p4_label, p3_label 中的
        # 一个位置中，该位置的 min_ciou_mask 为 True。
        # p5_label 形状为 (19, 19, 3, 85)，mask_p5 形状为 (19, 19, 3)，可以进行索引。
        p5_label[mask_p5] = object_box
        p4_label[mask_p4] = object_box
        p3_label[mask_p3] = object_box

    return p5_label, p4_label, p3_label


def _wrapper(one_image_path, dataset_type):
    """根据输入的图片路径，转换为 4 个 TF 张量返回。

    使用 wrapper 的目的，是在 dataset.map 和实际需要使用的函数之间做一个过渡，这样实际
    需要使用的函数内才能使用 eager 模式。

    使用方法：
    dataset = dataset.map(
        lambda one_image_path: wrapper(one_image_path, dataset_type),
        num_parallel_calls=tf.data.AUTOTUNE)

    Arguments:
        one_image_path： 一个字符串，图片存放路径。

    Returns:
        image_tensors: 一个图片的 3D 张量，形状为 (MODEL_IMAGE_HEIGHT, 
            MODEL_IMAGE_WIDTH, 3)。
        p5_label: 是一个 float32 型 4D 张量，形状为 (*FEATURE_MAP_P5, 3, 85)。
        p4_label: 是一个 float32 型 4D 张量，形状为 (*FEATURE_MAP_P4, 3, 85)。
        p3_label: 是一个 float32 型 4D 张量，形状为 (*FEATURE_MAP_P3, 3, 85)。
    """
 
    image_tensor, image_original_size = tf.py_function(
        func=_get_image_tensor_coco, inp=[one_image_path],
        Tout=(tf.float32, tf.float32))

    p5_label, p4_label, p3_label = tf.py_function(
        func=_get_label_arrays,
        inp=[one_image_path, image_original_size, dataset_type],
        Tout=(tf.float32, tf.float32, tf.float32))

    # 必须用 set_shape， 才能让 MapDataset 输出的张量有明确的形状。
    image_tensor.set_shape(shape=(*MODEL_IMAGE_SIZE, 3))
    # labels.set_shape(shape=3,)
    p5_label.set_shape(shape=[*FEATURE_MAP_P5, 3, 85])
    p4_label.set_shape(shape=[*FEATURE_MAP_P4, 3, 85])
    p3_label.set_shape(shape=[*FEATURE_MAP_P3, 3, 85])

    return image_tensor, p5_label, p4_label, p3_label


def _group_tensors_to_one(x, p5, p4, p3):
    """把 3 个 y 数组合并成一个。"""

    y = p5, p4, p3
    return x, y


def coco_data_yolov4_csp(dataset_type, images_range=(0, 1000),
                         shuffle_images=False, batch_size=8):
    """使用 COCO 2017 创建 tf.data.Dataset，训练 YOLO-V4-CSP 模型。

    使用方法：
    train_dataset = coco_data_yolov4_csp(
        dataset_type='validation', images_range=(0, None), batch_size=16)

    Arguments:
        dataset_type： 一个字符串，指定是训练集 'train' 或者验证集 'validation'。
        images_range： 一个元祖，分别表示开始的图片和最终的图片。例如 (10, 50) 表示从
            第 10 张图片开始，到第 49 张图片结束。None 或者 (0, None) 表示使用全部图片。
        shuffle_images： 一个布尔值，是否打乱图片顺序。
        batch_size： 一个整数，是批次量的大小。

    Returns:
        返回一个 tf.data.dataset，包含如下 2 个元素，image_tensors 和 labels：
            image_tensors: 一个图片的 float32 型张量，形状为
                (batch_size, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3)。
            labels 本身是一个元祖，它包含了 3 个 float32 型张量，分别为：
                p5_label: 形状为 (batch_size, *FEATURE_MAP_P5, 3, 85)。
                p4_label: 形状为 (batch_size, *FEATURE_MAP_P4, 3, 85)。
                p3_label: 形状为 (batch_size, *FEATURE_MAP_P3, 3, 85)。
    """

    # 设置好训练集图片和验证集图片的存放路径。
    path_image_train = (r'D:\deep_learning\computer_vision\COCO_datasets'
                        r'\COCO_2017\train2017')
    path_image_validation = (r'D:\deep_learning\computer_vision\COCO_datasets'
                             r'\COCO_2017\val2017')
    paths_image = {'train': path_image_train,
                   'validation': path_image_validation}

    # 获取训练集图片或是验证集图片的路径。
    path = paths_image.get(dataset_type, 'The input is invalid.')
    
    # image_paths 是一个列表，该列表包含了指定文件夹内，所有图片的完整路径。
    image_paths = _get_paths_image_coco(
        path_image=path, images_range=images_range,
        shuffle_images=shuffle_images)
    
    # 根据列表，生成 dataset。
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # wrapper 将返回一个图片张量，和 3 个标签张量 p5_label, p4_label, p3_label。
    dataset = dataset.map(
        lambda one_image_path: _wrapper(one_image_path, dataset_type),
        num_parallel_calls=tf.data.AUTOTUNE)

    # 前面 dataset 的各个方法，用于处理单个文件，下面用 batch 方法生成批量数据。
    # drop_remainder 为 True 表示如果最后一批的数量小于 BATCH_SIZE，则丢弃最后一批。
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # 把标签中的 3 个张量组合成一个元祖，目的是使得 dataset 的输出是 x，y 的形式，而不要
    # 出现 x, y1, y2, y3 的形式。Keras 的 fit 方法只接受 x，y 形式的 dataset。
    dataset = dataset.map(map_func=_group_tensors_to_one)

    # 预先取出一定数量的 dataset，用 tf.data.AUTOTUNE 自行调节数量大小。
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

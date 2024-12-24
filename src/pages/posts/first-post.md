---
layout: '@/templates/BasePost.astro'
title: 在RK3588上使用Yolov5
description: 机器视觉、深度学习
pubDate: 2024-11-21T00:00:00Z
imgSrc: '/assets/images/yolo.jpeg'
---
imgAlt: 'Image post 7'

友情链接[this page](../sixth-post/).

# 简介
因为xf车换了主控，所以需要重新部署一下模型在RK3588上面，写来给学弟们以后用

# 训练
### 选择你的版本
因为很多神秘的原因，Yolov5中我们不得不使用某一特定版本输出的权重文件才能够在最后一步推理的时候成功，所以这一步至关重要，需要这个仓库的内容。

airockchip/yolov5: YOLOv5 in PyTorch > ONNX > CoreML > TFLite (github.com)

值得注意的是你需要在这个仓库下训练。

关于环境安装的问题就不在此赘述了，这搞不懂后面你也看不懂。

## 预处理你的数据
对于xf比赛而言，我们并不需要模型具有良好的泛化性，换言之，可以尽情地使用各种数据增强的方法而完全不在乎过拟合的问题，故我们的输入只需要每个classes五张图片即可。
## 训练你的模型
在完成数据的生成之后，我们需要开始训练，请注意，你需要一张足够好的nvidia显卡，否则训练时间可能比你备赛的时间还要长。
首先需要创建一个配置文件，这里我随便命名一个
qwq.yaml

#### path: C://yolov5/auto_maker # dataset root dir
#### train: images # train images (relative to 'path') 90% of 847 train imagess
val: val/images # train images (relative to 'path') 10% of 847 train images

names:
#### 0: bodyarmor_1_5
#### 1: bodyarmor_2_5
#### 2: bodyarmor_3_5
#### 3: bodyarmor_4_5
#### 4: bodyarmor_5_5
#### 5: firstaid_1_5
#### 6: firstaid_2_5
#### 7: firstaid_3_5
#### 8: firstaid_4_5
#### 9: firstaid_5_5
#### 10: man_1_5
#### 11: man_2_5
#### 12: man_3_5
#### 13: smoke_1_5
#### 14: smoke_2_5
#### 15: smoke_3_5
#### 16: smoke_4_5
#### 17: smoke_5_5
#### 18: spontoon_1_5
#### 19: spontoon_2_5
#### 20: spontoon_3_5
#### 21: spontoon_4_5
#### 22: spontoon_5_5

推荐在这里对数据位置使用绝对路径，这样后期如果换别的网络也很方便继续使用同一份配置文件。其次是图像与labels的相对位置，验证集的位置和目标的种类，这个种类会在生成数据集的时候自动生成。

之后我们就可以开始训练了，以本例为例，我们稍微魔改一下train.py

        parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
        parser.add_argument('--batch-size', type=int, default=-1, help='total batch size for all GPUs, -1 for autobatch')
这样可以最大化利用你的显卡，顺便多练几轮狠狠地拟合。

之后在终端输入

        python .\train.py --data .\xView.yaml --weight yolov5n.pt --img 640

在训练完成后你就可以进入下一步了。
# 转换模型
## 导出为ONNX模型
就在训练的原工程下，输入这个，把位置换成你自己的权重文件就可以

        python .\export.py --rknpu --weight .\runs\train\exp2\weights\1.pt
## 转换为RKNN模型（这一步似乎不需要了）
这一步你需要在一台允许Ubuntu18/20/22的PC上进行，所以如果你没有的话，那你完了（btw其实xf车上也能干这事，看你自己了）

仓库地址在这里

https://github.com/airockchip/rknn_model_zoo
首先你还是需要安装必要的环境，这里不再赘述。
# 推理
首先还是需要在板子上面安装上面的这个库

https://github.com/airockchip/rknn_model_zoo
之后把模型和以下文件放在一起：

        import os
        import cv2
        import sys
        import argparse
        import time
        /# add path
        realpath = os.path.abspath(__file__)
        _sep = os.path.sep
        realpath = realpath.split(_sep)
        sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('scripts')+1]))
        
    from py_utils.coco_utils import COCO_test_helper
    import numpy as np
    
    
    OBJ_THRESH = 0.25
    NMS_THRESH = 0.45
    
    /# The follew two param is for map test
    /# OBJ_THRESH = 0.001
    /# NMS_THRESH = 0.65
    
    IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)
    
    
    
    /# modefile the things there
    CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
    "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
    "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
    "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
    "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
    
    coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    
    def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

    def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    /# Returns
    keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


    def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy

    def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    /# 1*255*h*w -> 3*85*h*w
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
    boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
    scores.append(input_data[i][:,4:5,:,:])
    classes_conf.append(input_data[i][:,5:,:,:])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


    def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
    top, left, right, bottom = [int(_b) for _b in box]
    print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
    cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
    cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
    platform = 'pytorch'
    from py_utils.pytorch_executor import Torch_model_container
    model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
    platform = 'rknn'
    from py_utils.rknn_executor import RKNN_model_container
    model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
    platform = 'onnx'
    from py_utils.onnx_executor import ONNX_model_container
    model = ONNX_model_container(args.model_path)
    else:
    assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform
    
    def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
    if path.endswith(_type) or path.endswith(_type.upper()):
    return True
    return False
    
    class Detect:
    def __init__(self) -> None:
    /# 初始化摄像头
    self.cap = cv2.VideoCapture(0)

        parser = argparse.ArgumentParser(description='Process some integers.')
        # basic params
        parser.add_argument('--model_path', type=str, default= "/home/iflytek/ucar_ws/src/ucar_nav/scripts/scripts/1.onnx",  help='model path, could be .pt or .rknn file')
        parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
        parser.add_argument('--device_id', type=str, default=None, help='device id')
        
        parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
        parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

        # data params
        parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
        # coco val folder: '../../../datasets/COCO//val2017'
        parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
        parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')
        parser.add_argument('--anchors', type=str, default='/home/iflytek/ucar_ws/src/ucar_nav/scripts/scripts/anchors_yolov5.txt', help='target to anchor file, only yolov5, yolov7 need this param')

        args = parser.parse_args()

        # load anchor
        with open(args.anchors, 'r') as f:
            values = [float(_v) for _v in f.readlines()]
            self.anchors = np.array(values).reshape(3,-1,2).tolist()
        print("use anchors from '{}', which is {}".format(args.anchors, self.anchors))
        
        # init model
        self.model, self.platform = setup_model(args)

        file_list = sorted(os.listdir(args.img_folder))
        img_list = []
        for path in file_list:
            if img_check(path):
                img_list.append(path)
        self.co_helper = COCO_test_helper(enable_letter_box=True)

        # run test
        for i in range(len(img_list)):
            print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

            img_name = img_list[i]
            img_path = os.path.join(args.img_folder, img_name)
            if not os.path.exists(img_path):
                print("{} is not found", img_name)
                continue


    def get_data(self):
        ret, img_src = self.cap.read() 
        # 在这里开始魔改
        if not ret:
            return None
        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0,0,0)
        img = self.co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # preprocee if not rknn model
        if self.platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img

        outputs = self.model.run([input_data])
        boxes, classes, scores = post_process(outputs, self.anchors)
        print(boxes,classes,scores)
        return boxes, classes, scores
    def free_cam(self):
        self.cap.release()
        return 1
    if __name__ == "__main__":
    det = Detect()
    while True:
    det.get_data()
之后在终端输入
sudo modprobe rknpu
更改默认的模型目录，直接运行即可。
import torch

from aim_tools.config import *
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.torch_utils import select_device
import numpy as np
from utils.general import  check_img_size, \
    non_max_suppression, scale_coords,  \
    xyxy2xywh
from utils.plots import Annotator, colors

# def getModel(img0):
#     #cpu or jpu
#     device = select_device('')
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#     stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
#     model = attempt_load(WEIGHTS, map_location=device)
#     stride = int(model.stride.max())  # model stride
#     names = model.module.names if hasattr(model, 'module') else model.names  # get class names
#     if half:
#         model.half()  # to FP16
#     return model, device, half, stride, names

    # imgsz = check_img_size(IMGSZ, s=stride)  # check image size
    # # Padded resize
    # img = letterbox(img0, IMGSZ, stride=stride, auto=True)[0]
    #
    # # Convert
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # img = np.ascontiguousarray(img)
    #
    # bs = 1  # batch_size
    # vid_path, vid_writer = [None] * bs, [None] * bs
    # model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    # img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    # if len(img.shape) == 3:
    #     img = img[None]  # expand for batch dim
    # pred = model(img, augment=False, visualize=False)[0]
    # pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=MAX_DET)
    # det = next(pred)
    # im0 = img0.copy()
    # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # imc = im0.copy() if save_crop else im0  # for save_crop
    # annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))
    # if len(det):
    #     # Rescale boxes from img_size to im0 size
    #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #     # Write results
    #     for *xyxy, conf, cls in reversed(det):
    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #
    #         c = int(cls)  # integer class
    #         label = None if HIDE_LABEL else (names[c] if HIDE_CONF else f'{names[c]} {conf:.2f}')
    #         annotator.box_label(xyxy, label, color=colors(c, True))
    # im0 = annotator.result()
    # if view_img:
    #     cv2.imshow(str(p), im0)
    #     cv2.waitKey(1)  # 1 millisecond

def getModel():
    #cpu or jpu
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    model = attempt_load(WEIGHTS, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    return model, device, half, stride, names



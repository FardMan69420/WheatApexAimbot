import mss
import numpy as np
import cv2
import pyautogui
from pynput import mouse

from aim_tools.mouse import move_mouse
from main import *
import pynput
sct = mss.mss();
screen_width = 1920
screen_height = 1080
game_left, game_top, game_width, game_height = screen_width // 3, screen_height // 3, screen_width // 3, screen_height // 3
reWinWidth, reWinHeight = screen_width // 5, screen_height // 5
LOCK_AIM  = False
monitor = {
    'left': game_left,
    'top': game_top,
    'width': game_width,
    'height': game_height
}
windowName = 'test'

model, device, half, stride, names = getModel()
mouseControl = pynput.mouse.Controller()
imgsz = check_img_size(IMGSZ, s = stride)  # check image size\




def mouseAim(xywh_list, mouse, left, top, width, height):
    mouseX, mouseY = mouse.position
    best_xy = None
    for xywh in xywh_list:
        x, y, _, _ = xywh
        x *= width
        y *= height
        x += left
        y += top
        dist = ((x - mouseX) ** 2 + (y - mouseY) ** 2) ** 0.5  # 欧氏距离
        if not best_xy:
            best_xy = ((x, y), dist)
        else:
            _, old_dist = best_xy
            if dist<old_dist:
                best_xy =  ((x, y), dist)

    x, y = best_xy[0]

    pyautogui.moveTo(x, y)#pyautogui.moveTo(1800, 500, duration=2, tween=pyautogui.easeInOutQuad)
def on_click(x, y, button, pressed):
    global LOCK_AIM
    if button == button.x2:
        if pressed:
            LOCK_AIM = not LOCK_AIM
            print('自瞄状态：', pressed)
listener = mouse.Listener(on_click=on_click)
listener.start()







@torch.no_grad()
def pred_img(img0):
    # Padded resize
    img = letterbox(img0, IMGSZ, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=MAX_DET)

    _, det = next(enumerate(pred))
    im0 = img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))
    xywh_list = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywh_list.append(xywh)
            label = None if HIDE_LABEL else (names[c] if HIDE_CONF else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    return im0, xywh_list
    # if view_img:
    #     cv2.imshow(str(p), im0)
    #     cv2.waitKey(1)  # 1 millisecond


while True:
    img = sct.grab(monitor=monitor)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img, aims = pred_img(img)
    if aims and LOCK_AIM:
        mouseAim(aims, mouseControl, **monitor)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, reWinWidth, reWinHeight)
    cv2.imshow(windowName,img)
    k = cv2.waitKey(1)
    if k%256 == 27:
        cv2.destroyAllWindows()
        exit('结束')


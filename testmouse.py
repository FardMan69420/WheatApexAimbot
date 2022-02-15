import time
import pyautogui
import pynput # 需要pip install一下
from aim_tools.mouse import move_mouse, gm

x_true, y_true=500, 500# 准备移动到的点
mouseControl = pynput.mouse.Controller()# 创建控制对象
while True:
    x_mouse, y_mouse = mouseControl.position
    print('鼠标位置',x_mouse, y_mouse)
    print(x_true - x_mouse, y_true - y_mouse)
    pyautogui.moveTo(500,500)
    # move_mouse(10,10)
    time.sleep(1)
    #move_mouse是你输入多少，鼠标就动多少，比如一直重复50 50就会去右下角
    #理想情况应该是：
    #想去的点-现在的鼠标位置 = 需要移动的距离
    #使用move_mouse来移动刚刚获得的值
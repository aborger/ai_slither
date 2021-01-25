from PIL import ImageGrab, ImageOps
import os
import time
import numpy as np
import win32api, win32con

XPAD = 0
YPAD = 129

 
def LeftDown():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(.1)
    print("Left Down")

def LeftUp():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    time.sleep(.1)
    print("Left Up")

def mousePos(cord):
    win32api.SetCursorPos((XPAD + cord[0], YPAD + cord[1]))

def get_cords():
    x, y = win32api.GetCursorPos()
    x = x - XPAD
    y = y - YPAD
    print (x, y)

def screenGrab():
    box = (XPAD + 1, YPAD + 1, XPAD + 1919, YPAD + 900)
    im = ImageGrab.grab(bbox=box)
    im = im.resize((480, 225))
    im = ImageOps.grayscale(im)
    im = ImageOps.autocontrast(im, cutoff=10)

    im_data = np.asarray(im)
    im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) +
'.png', 'PNG')
 
def main():
    screenGrab()
 
if __name__ == '__main__':
    main()
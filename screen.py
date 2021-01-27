from PIL import ImageGrab, ImageOps
import os
import time
import numpy as np


XPAD = 0
YPAD = 129

SCORE_XPAD = 107
SCORE_YPAD = 983

TRUTH_DATA_FILE = 'Score_data_truth.csv'


def screenGrab():
    game_box = (XPAD + 1, YPAD + 1, XPAD + 1919, YPAD + 900)
    score_box = (SCORE_XPAD + 1, SCORE_YPAD + 1, SCORE_XPAD + 45, SCORE_YPAD + 16)

    gameIm = ImageGrab.grab(bbox=game_box)
    scoreIm = ImageGrab.grab(bbox=score_box)

    
    #gameIm = gameIm.resize((480, 225))
    gameIm = ImageOps.grayscale(gameIm)
    scoreIm = ImageOps.grayscale(scoreIm)

    game_data = np.asarray(gameIm, dtype=np.float64)
    score_data = np.asarray(scoreIm, dtype=np.float64)

    return (gameIm, scoreIm, game_data, score_data)

def saveData():
    screen = screenGrab()
    np.save(os.getcwd() + '/score_data/' + str(int(time.time())), screen[3])
 


def saveImage():
    screen = ImageGrab.grab(bbox=())
    screen.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) +'.png', 'PNG')

def populateCSV():
    data_file = open(os.getcwd() + '/' + TRUTH_DATA_FILE, 'w')
    data_file.write('File, Value\n')
    score_data = os.listdir(os.getcwd() + '/score_data/')
    for score in score_data:
        data_file.write(score + '\n')

def record():
    while True:
        data_num = int(time.time())
        print("Data!")
        screen = screenGrab()
        np.save(os.getcwd() + '/score_data/' + str(data_num), screen[3])
        screen[1].save(os.getcwd() + '/score_image/' + str(data_num) + '.png', 'PNG')
        data_num += 1
        time.sleep(2)


def main():
    saveImage()
 
if __name__ == '__main__':
    main()
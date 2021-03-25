from PIL import ImageGrab, ImageOps, Image
import os
import time
import numpy as np


XPAD = 0
YPAD = 129

SCORE_XPAD = 107
SCORE_YPAD = 983

TRUTH_DATA_FILE = 'Score_data_truth.csv'
NUM_SCORE_DIGITS = 5
game_box = (XPAD + 1, YPAD + 1, XPAD + 1919, YPAD + 900)
score_box = (SCORE_XPAD + 1, SCORE_YPAD + 1, SCORE_XPAD + 45, SCORE_YPAD + 16)

def screenGrab():

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
    screen = ImageGrab.grab(bbox=score_box)
    screen.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) +'.png', 'PNG')

def showImage():
    img = Image.open('scoreImg.png')
    DIGIT_WIDTH = 9
    for dig in range(0, NUM_SCORE_DIGITS):
        cropped = img.crop((dig * DIGIT_WIDTH, 0, (dig + 1) * DIGIT_WIDTH, 15))
        cropped.show()


def populateCSV():
    data_file = open(os.getcwd() + '/' + TRUTH_DATA_FILE, 'w')
    data_file.write('File, Value\n')
    score_data = os.listdir(os.getcwd() + '/score_image/')
    for score in score_data:
        img = Image.open(os.getcwd() + '/score_image/' + score)
        img.show()
        real_score = input("Enter Score: ")
        data_file.write(score + ', ' + real_score + '\n')

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
    record()
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Control Trading AI')
    parser.add_argument("command", metavar="<command>", help="'record', 'populate'")
	
    """
    Examples:
    parser.add_argument("-t", action='store_true', required=False,
						help= "Include -t if this is a shortened test")
						
    parser.add_argument("--name", help = "Name for new model when training")
    """
						
			
    args = parser.parse_args()

    if args.command == 'record':
        record()
    elif args.command == 'populate':
        populateCSV()
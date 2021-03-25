import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
import time

NUM_SCORE_DIGITS = 5
IMAGE_DIR = '/score_image/'
DATA_DIR = '/score_ai_training/'
TRAINING_DIR = DATA_DIR + 'training/'
VALIDATION_DIR = DATA_DIR + 'validation/'
MODEL_DIR = './models'
SCORE_IMG_SHAPE = (1, 9, 15)


class config:
    optimizer = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()
    epochs = 50


def ScoreDigits(score_img):
    digit_imgs = []
    for dig in range(0, NUM_SCORE_DIGITS):
            cropped = score_img.crop((dig * 9, 0, (dig + 1) * 9, 15))
            digit_imgs.append(cropped)
    return digit_imgs


def scale(data, nmin = 0, nmax = 1):
    minimum = np.amin(data)
    maximum = np.amax(data)

    new_data = []
    for x in data:
        new_x = []
        for y in x:
            new_x.append((y - minimum) * (nmax - nmin) / (maximum - minimum) + nmin)
        new_data.append(new_x)
    return new_data
    

class ScoreKeeper(tf.keras.Model):
    def __init__(self):
        super(ScoreKeeper, self).__init__()

        self.reshape = tf.keras.layers.Reshape((135,), input_shape=(SCORE_IMG_SHAPE))
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(11, 'softmax')
        


    def call(self, input):
        x = self.reshape(input)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def __train_step(self, input, correct_output):
        with tf.GradientTape() as tape:
            pred = self.call(input)
            # Create one hot
            true = tf.zeros([11]).numpy()
            if correct_output == '!':
                correct_output = 10
            else:
                correct_output = int(correct_output)
            true[correct_output] = 1
            loss = config.mse(true, pred)
        grads = tape.gradient(loss, self.trainable_variables)
        config.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def train(self):
        truth_data = pd.read_csv(os.getcwd() + DATA_DIR + 'training_truth.csv', index_col='File', sep=r'\s*,\s*', engine='python')
        training_data = os.listdir(os.getcwd() + TRAINING_DIR)
        for score_file in training_data:
            truth = truth_data.loc[score_file, 'Value']
            data = np.load(os.getcwd() + TRAINING_DIR + score_file)
            img = Image.fromarray(data)

            # contains images of individual digits
            digit_imgs = ScoreDigits(img)


            truth_str = str(truth)
            while len(truth_str) < 5:
                truth_str += '!'

            if len(truth_str) != len(digit_imgs) or len(truth_str) != NUM_SCORE_DIGITS:
                raise ValueError("Truths and digits are not the same")
            
           
            for digit in range(0, NUM_SCORE_DIGITS):
                # convert image data into text data
                digit_data = np.asarray(digit_imgs[digit], dtype=np.float32)
                scaled = scale(digit_data)
                reshaped = np.reshape(scaled, SCORE_IMG_SHAPE)
                self.__train_step(reshaped, truth_str[digit])

    def getScore(self, img):
        digit_imgs = ScoreDigits(img)
        predicted_digits = []
        for digit in digit_imgs:
            digit_data = np.asarray(digit, dtype=np.float32)
            scaled = scale(digit_data)
            reshaped = np.reshape(scaled, SCORE_IMG_SHAPE)
            output = self.predict(reshaped)
            output = np.argmax(output)
            if(output != 10):
                predicted_digits.append(output)

        score_str = ''
        for digit in predicted_digits:
            score_str += str(digit)
        if score_str == '':
            return 0
        else:
            score = int(score_str)
            return score






def train():
    ai = ScoreKeeper()

    truth_data = pd.read_csv(os.getcwd() + DATA_DIR + 'validation_truth.csv', index_col='File', sep=r'\s*,\s*', engine='python')
    validation_data = os.listdir(os.getcwd() + VALIDATION_DIR)
    epoch = 0
    done = 0
    while epoch < config.epochs and done < 2:
        accuracy_sum = 0
        ai.train()

        #validate
        for score_file in validation_data:
                truth = int(truth_data.loc[score_file, 'Value'])
                data = np.load(os.getcwd() + VALIDATION_DIR + score_file)
                img = Image.fromarray(data)
                score = ai.getScore(img)

                if truth == score:
                    accuracy_sum += 1
                else:
                    print('truth: ' + str(truth) + ' score: ' + str(score))

        print('Epoch: ' + str(epoch) + ' of ' + str(config.epochs))
        print('Accuracy: ' + str(accuracy_sum / len(validation_data)))
        if int(accuracy_sum / len(validation_data)) > 90:
            done += 1
        epoch += 1

    ai.save_weights(filepath=MODEL_DIR + '/scoreModel2')

            

 
if __name__ == '__main__':
    train()
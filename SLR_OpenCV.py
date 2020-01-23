import cv2
import numpy as np
from keras.models import load_model
from time import time

model = load_model('SLR_CNN_colab.h5')


def prediction(pred_class):
    return (chr(pred_class + 65))


def keras_pred(model, image):

    data = np.array(image, dtype='uint8')
    pred_prob = model.predict(data)[0]
    pred_class = list(pred_prob).index(max(pred_prob))

    return max(pred_prob), pred_class


# def crop_img(img, x, y, width, height):
#     return img[y:y + height, x:x + width]

def main():
    count = 0
    previous = time()
    delta = 0
    while True:
        capture = cv2.VideoCapture(0)
        _, frame = capture.read()

        cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)
        im2 = frame[100:350, 100:350]

        img_gauss = cv2.GaussianBlur(im2, (5, 5), 0)

        img_gray = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)

        im3 = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        im4 = np.resize(im3, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)

        current = time()
        delta += current - previous
        previous = current

        # Check if 3 (or some other value) seconds passed
        if delta > 3:
            count += 1
            cv2.imwrite("img/frame{}.jpg".format(count), im4)
            pred_prob, pred_class = keras_pred(model, im5)

            curr = prediction(pred_class)

            cv2.putText(frame, curr, (100, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            # Reset the time counter
            delta = 0

        # cv2.rectangle(frame, (200, 200), (500, 500), (0, 255, 0), 0)
        cv2.imshow("Frame", frame)
    #     cv2.imshow("Blurred Gray", img_gauss)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            capture.release()
            break


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()

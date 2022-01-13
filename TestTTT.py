import cv2
import numpy as np
from scipy.stats import itemfreq
import pickle

###################################################33

# #######################################3
font = cv2.FONT_HERSHEY_SIMPLEX
from tensorflow import keras
my_model = keras.models.load_model("my_model")
my_model.load_weights("weights.h5")

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):
    if classNo == 0: return 'Stop'
    elif classNo == 1: return 'Right'
    elif classNo == 2: return 'Left'
    elif classNo == 3: return 'ahead only'

##################################33


######################################################



clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0) 
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse)

success, frame = cameraCapture.read()

while success and not clicked:
    cv2.waitKey(1)
    success, frame = cameraCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 37)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=120, param2=40)

    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = frame[y-r:y+r, x-r:x+r]
            img = np.asarray(square)
            img = cv2.resize(img, (32,32))
            img = preprocessing(img)
            cv2.imshow("Processed Image",img)
            img = img.reshape(1,32,32,1)
            cv2.putText(square, "TRAFFIC SIGN: ",(20,35),font,0.75, (0,0,255),2,cv2.LINE_AA)
            cv2.putText(square, "PROBABILITY: ",(20,75),font,0.75, (0,0,255),2,cv2.LINE_AA)
        #   PREDICT IMAGE
            predictions = my_model.predict(img)
            classIndex = my_model.predict_classes(img)
            probabilityValue = np.amax(predictions)
            if probabilityValue > 0.99:
                print(str(getClassName(classIndex)))
                cv2.imshow("Result",square)
            
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cameraCapture.release()

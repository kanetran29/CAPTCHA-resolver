import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model('best_model.keras')

IMG_HEIGHT = 40
IMG_WIDTH = 150  
CHAR_SET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' 
CAPTCHA_LENGTH = 5  

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255.0              
    img = np.expand_dims(img, axis=-1)               
    img = np.expand_dims(img, axis=0)                 
    return img

def decode_prediction(pred_list):
    captcha_text = ''
    for pred in pred_list:
        char_index = np.argmax(pred[0])
        captcha_text += CHAR_SET[char_index]
    return captcha_text

def predict_captcha(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    captcha = decode_prediction(prediction)
    return captcha

def show_image(img_path):
    img = cv2.imread(img_path)
    cv2.imshow("CAPTCHA Image", img)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image = 'data/test1.png'
    # show_image(test_image)
    captcha_text = predict_captcha(test_image)
    print("Predicted CAPTCHA:", captcha_text)

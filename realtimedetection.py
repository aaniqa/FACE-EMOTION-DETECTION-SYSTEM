import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()


model = model_from_json(model_json, custom_objects={
    'Sequential': Sequential,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Dropout': Dropout,
    'Flatten': Flatten,
    'Dense': Dense,
    'InputLayer': InputLayer
})

# Load the model weights
model.load_weights("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    # Convert the image to numpy array and preprocess
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Set up the webcam for real-time emotion detection
webcam = cv2.VideoCapture(0)

# Define labels for the emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Capture frame from webcam
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            # Extract the region of interest (ROI) of the face
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))  # Resize to match model input size
            img = extract_features(image)
            
            # Predict emotion
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]  # Get the emotion label from the prediction
            
            # Display the predicted emotion on the image
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Show the output image with the detected face and emotion label
        cv2.imshow("Output", im)
        
        # Break the loop on pressing 'Esc' key
        if cv2.waitKey(27) & 0xFF == 27:  # 27 is the ASCII code for 'Esc' key
            break
    except cv2.error:
        pass

# Release webcam and close OpenCV window
webcam.release()
cv2.destroyAllWindows()

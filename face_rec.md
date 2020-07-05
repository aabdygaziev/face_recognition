# Face Recognition Algorithm with OpenCV

Today we will try to build face recognition algorithm that tells you who is in fron of the camera. To do that, we need sample of training data. For the simplicity, I used my own pictures, and dowloanded from internet pictures of GOT stars - Kit Harrington, and Lena Headey.

This project is inspired from <a href='https://www.codingforentrepreneurs.com/'> codingforentrepreneurs youtube channel</a>.

Before starting make sure that you have opencv installed.

## Building the model

First, we need to train our model on sample traingin data so it can recognize faces later on.


```python
# import packages
import os
from _testcapi import DBL_MAX

import numpy as np
import cv2
from PIL import Image
import pickle
```


```python
# define path to data-set

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'images')
```


```python
# define our face-detection classifier

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# creating face recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create(
    neighbors=8,
    radius=1,
    grid_x=8,
    grid_y=8,
    threshold=DBL_MAX
)

# captures frames from our webcamer (0). 
# 0 means your first webcamera
cap = cv2.VideoCapture(0) 
```

There are other types of face recognizer model such as FisherFaces, and Eigen Faces. You can learn more about them here on this <a href='https://github.com/rragundez/PyData/blob/master/notebooks_tutorial/03_Building_the_Recognition_Model.ipynb'>repo by Rodrigo Agundez</a>.

You can download cascade classifier from <a href='https://github.com/opencv/opencv/tree/master/data/haarcascades'>opencv github repo</a>. There several haarcascade classifiers, but we need **'haarcascade_frontalface_default.xml'**

Now let's upload our data-set. We need to define labels, and traning data set.


```python
current_id = 0
label_ids = {}
y_labels = []
x_train = []
```


```python
# uploading data set
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("JPG") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            
            pil_image = Image.open(path).convert("L")  # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)
```


```python
# save labels as pickle file
with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)
```

## Training the model


```python
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')
```

Now we have finished building and training our model. Next, we will write our face detection algorithm. We saved our trained model as '**trainer.yml**'.

**Note**: all of this codes are writen on PyCharm. 

## Model implementation


```python
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
cap.set(cv2.CAP_PROP_FPS, 25)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')


with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=2,
        minNeighbors=5
    )
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        # roi stands for 'region of interest'
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # recognize?
        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(
                frame, name,
                (x, y), font,
                1, color,
                stroke, cv2.LINE_AA)
        # save img
        img_item = 'my_image.png'
        cv2.imwrite(img_item, roi_gray)

        # draw rectangle
        # BGR color
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(
            frame,
            (x, y),
            (end_cord_x, end_cord_y),
            color,
            stroke
        )
    # Display the resulting time
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

## Results

The model is working, but it's not identifing a person correctly. The reason is data. I have very small data set, and quality of pictures also  affecting the model. You can work with this model and try it. It's fun!


```python

```

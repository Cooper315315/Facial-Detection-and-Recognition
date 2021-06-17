import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
import os
import cv2
from tensorflow.keras import callbacks

#import data
from os import listdir
path = r"C:/Users/User/Documents/FTDS/FTDS student material/COLAB_project/data/UTKFace/"
files = os.listdir(path)

#load images
import random
images = []
ages = []
genders = []
for file in random.sample(files, 5000):
    if len(file.split('_'))==4: 
        age = int(file.split('_')[0])
        if age > 80:
            age = '10'
        elif age > 60:
            age = '9'
        elif age > 47:
            age = '8'
        elif age > 37:
            age = '7'
        elif age > 31:
            age = '6'
        elif age > 22:
            age = '5'
        elif age > 16:
            age = '4'
        elif age > 10:
            age = '3'
        elif age > 5:
            age = '2'
        elif age > 2:
            age = '1'
        else:
            age = '0'
        ages.append(age)
        genders.append(int(file.split('_')[1]))
        image = cv2.imread(path+file)
        image = cv2.resize(image,dsize=(64,64))
        images.append(image)

# EDA
# age distribution
x_ages = list(set(ages))
y_ages = [ages.count(i) for i in x_ages]
plt.bar(x_ages,y_ages)
plt.show()
print("Max value:",max(ages))

# gender distribution
x_genders = list(set(genders))
y_genders = [genders.count(i) for i in x_genders]
plt.bar(x_genders,y_genders)
plt.show()

# gender prediction model
# train test split
X=np.array(images)
y_gender=np.array(genders)
X_train, X_test, y_train, y_test=train_test_split(X,y_gender,test_size=0.2,random_state=42)
X_train=X_train/255.
X_test=X_test/255.

# build gender model
gender_model=Sequential()
gender_model.add(Conv2D(32,(3, 3),activation='relu', input_shape=(64,64,3)))
gender_model.add(Conv2D(32,(3, 3),activation='relu'))
gender_model.add(MaxPooling2D(2, 2))
gender_model.add(Conv2D(32,(3, 3),activation='relu'))
gender_model.add(Conv2D(32,(3, 3),activation='relu'))
gender_model.add(MaxPooling2D(2, 2))
gender_model.add(Conv2D(64,(3, 3),activation='relu'))
gender_model.add(Conv2D(64,(3, 3),activation='relu'))
gender_model.add(MaxPooling2D(2, 2))
gender_model.add(Dropout(0.25))
gender_model.add(Flatten())

gender_model.add(Dense(128, activation='relu'))
gender_model.add(Dense(64, activation='relu'))
gender_model.add(Dense(32, activation='relu'))
gender_model.add(Dropout(0.5))
gender_model.add(Dense(1, activation='sigmoid'))
gender_model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

early_stop=callbacks.EarlyStopping(monitor='val_loss', patience=3)
hist=gender_model.fit(X_train, y_train, epochs=20, batch_size=20, validation_data=(X_test, y_test),callbacks=[early_stop])

# gender model summary
gender_model.summary()

#plot Training Accuracy and Validation Accuracy
epoch_list=list(range(1, len(hist.history["accuracy"])+1))
plt.plot(epoch_list, hist.history["accuracy"], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy','Validation Accuracy'))
plt.show()

# age prediction model
# train test split
X=np.array(images)
y_age=np.array(ages)
X_train, X_test, y_train, y_test=train_test_split(X,y_age,test_size=0.2,random_state=42)
X_train=X_train/255.
X_test=X_test/255.

# categorical 10 classe
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

# build age model
age_model=Sequential()
age_model.add(Conv2D(32,(3, 3),activation='relu', input_shape=(64,64,3)))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Conv2D(64,(3, 3),activation='relu'))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Conv2D(128,(3, 3),activation='relu'))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Conv2D(256,(3, 3),activation='relu'))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Dropout(0.25))
age_model.add(Flatten())

age_model.add(Dense(128, activation='relu'))
age_model.add(Dense(64, activation='relu'))
age_model.add(Dense(32, activation='relu'))
age_model.add(Dropout(0.5))
age_model.add(Dense(10, activation='softmax'))
age_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss ='categorical_crossentropy', metrics=['accuracy'])

early_stop=callbacks.EarlyStopping(monitor='val_loss', patience=3)
hist=age_model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_test, y_test),callbacks=[early_stop])

# age model summary
gender_model.summary()

# plot Training Accuracy and Validation Accuracy
epoch_list=list(range(1, len(hist.history["accuracy"])+1))
plt.plot(epoch_list, hist.history["accuracy"], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy','Validation Accuracy'))
plt.show()



#################emo model############################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense,Dropout
from keras import backend as K
from pathlib import Path
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

train_dir = "train" #passing the path with training images
test_dir = "test"   #passing the path with testing images
img_size = 48 #original size of the image

# Apply ImageData Generator
train_datagen = ImageDataGenerator(rotation_range = 22,
                                         width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         #zoom_range = 0.2,
                                         validation_split = 0.2
                                        )
validation_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                    target_size = (img_size,img_size),
                                                    batch_size = 64,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                    subset = "training"
                                                   )
validation_generator = validation_datagen.flow_from_directory( directory = test_dir,
                                                              target_size = (img_size,img_size),
                                                              batch_size = 64,
                                                              color_mode = "grayscale",
                                                              class_mode = "categorical",
                                                              subset = "validation"
                                                             )
# build emotion model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer = Adam(lr=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 15
batch_size = 64
history = model.fit(x = train_generator,epochs = epochs,validation_data = validation_generator)

fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

# Review accuracy graphs
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Save JSON and h5
# from pathlib import Path
# model_structure = model.to_json()
# f = Path("EMO_MODEL_STRUCTURE.json")
# f.write_text(model_structure)
# model.save_weights("EMO_MODEL_WEIGHTS.h5")

# Make prediction
img = image.load_img("test/neutral/im14.png",target_size = (48,48),color_mode = "grayscale")
img = np.array(img)
plt.imshow(img)
# print(img.shape)
label_dict = {0:'Disgust',1:'Happy',2:'Neutral'}
img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
img = img.reshape(1,48,48,1)
result = model.predict(img)
result = list(result[0])
print(result)
img_index = result.index(max(result))
print(label_dict[img_index])
plt.show()
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc   = model.evaluate(validation_generator)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))

# face detection and prediction model
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path
from tensorflow.keras.models import model_from_json

# load gender model
f=Path('cnn_gender_rgb.json')
model_structure=f.read_text()
gender_model=model_from_json(model_structure)
gender_model.load_weights('cnn_gender_rgb.h5')
# load age model
f=Path('cnn_age10_rgb.json')
model_structure=f.read_text()
age_model=model_from_json(model_structure)
age_model.load_weights('cnn_age10_rgb.h5')
# load emotion model
f=Path('EMO_MODEL_STRUCTURE.json')
model_structure=f.read_text()
emo_model=model_from_json(model_structure)
emo_model.load_weights('EMO_MODEL_WEIGHTs.h5')


age_labels=['0-2','3-11', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-80', '81-116']
emo_dict = {0:'Disgust',1:'Happy',2:'Neutral'}

# face detection function
def face_detection(path, faces):
  data=pyplot.imread(path)
  pyplot.imshow(data)
  fig=pyplot.gcf()
  fig.set_size_inches(15,10)
  ax=pyplot.gca()
  for face in faces:
    x,y,width,height=face['box']

    # center alignment 
    center=[x+(width/2),y+(height/2)]
    max_border=max(width,height)
    left=max(int(center[0]-(max_border/2)),0)
    right=max(int(center[0]+(max_border/2)),0)
    top=max(int(center[1]-(max_border/2)),0)
    bottom=max(int(center[1]+(max_border/2)),0)

    # crop the face
    cropped_image_i=data[top:top+max_border,left:left+max_border,:]
    # resize image to fit the age and gender model
    cropped_image=np.array(Image.fromarray(cropped_image_i).resize([64,64]))
    # resize image to fit emo model
    cropped_image_emo=np.array(Image.fromarray(cropped_image_i).resize([48,48]))
    cropped_image_emo=cv2.cvtColor(cropped_image_emo,cv2.COLOR_BGR2GRAY)
    cropped_image_emo=np.expand_dims(cropped_image_emo,0)

    # create predictions
    gender_pred=gender_model.predict(cropped_image.reshape(1,64,64,3))
    age_pred=age_model.predict(cropped_image.reshape(1,64,64,3))
    emo_pred=emo_model.predict(cropped_image_emo.reshape(1,48,48,1))

    # create the box around the face
    rect=Rectangle((left,top),max_border,max_border,fill=False,color='red')
    # add the box
    ax.add_patch(rect)
    
    # add gender prediction
    gender_text='Female' if gender_pred > 0.5 else 'Male'
    ax.text(left,top-(image.shape[0]*0.014),'Gender: {}'.format(gender_text),fontsize=10,color='red')
    # # add age prediction
    age_index=int(np.argmax(age_pred))
    age_confident=age_pred[0][age_index]
    age_text=age_labels[age_index]
    ax.text(left,top-(image.shape[0]*0.033),'Age: {}({:.2f})'.format(age_text,age_confident),fontsize=10,color='red')
    # add emotion prediction
    emo_index=np.argmax(emo_pred)
    emo_confident=np.max(emo_pred)
    emo_text=emo_dict[emo_index]
    ax.text(left, top-(image.shape[0]*0.055),'Emotion: {}({:.2f})'.format(emo_text,emo_confident),fontsize=10,color='red')
  # show the resulting image
  pyplot.show()

path='./face_detection_images/sadman.jpg'
image=pyplot.imread(path)
detector=MTCNN()
faces=detector.detect_faces(image)
face_detection(path,faces)

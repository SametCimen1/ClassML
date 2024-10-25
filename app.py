import os
import time 
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
import warnings
warnings.filterwarnings("ignore")


class EyeDiseaseDataset:
    def __init__(self, dataDir):
            self.data_dir = dataDir
    
    def dataPaths(self):
            filepaths = []
            labels = []
            folds = os.listdir(self.data_dir)
            for fold in folds:
                foldPath = os.path.join(self.data_dir, fold)
                filelist = os.listdir(foldPath)
                for file in filelist:
                    fpath = os.path.join(foldPath, file)
                    filepaths.append(fpath)
                    labels.append(fold)
            return filepaths, labels
  
    def dataFrame(self, files, labels):
            Fseries = pd.Series(files, name='filepaths')
            Lseries = pd.Series(labels, name='labels')
            return pd.concat([Fseries, Lseries], axis=1)
    
    def split_(self):
            files, labels = self.dataPaths()
            df = self.dataFrame(files, labels)
            strat = df['labels']
            trainData, dummyData = train_test_split(df, train_size=None, shuffle=True, random_state=42, stratify=strat)
            strat = dummyData['labels']
            validData, testData = train_test_split(dummyData, train_size=None, shuffle=True, random_state=42, stratify=strat)
            return trainData, validData, testData

dataDir="C:\\Users\\cimen\\Desktop\\code\\tenserflow\\content\\dataset"


dataSplit = EyeDiseaseDataset(dataDir)
train_data, valid_data, test_data = dataSplit.split_()

print("TEST DATA")
print(test_data)

def augment_data( train_df, valid_df, test_df, batch_size=32):      
  img_size = (256,256)
  channels = 3
  color = 'rgb'

  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
              rotation_range=30,
              horizontal_flip=True,
              vertical_flip=True,
              brightness_range=[0.5, 1.5])
          
  valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
          
  train_generator = train_datagen.flow_from_dataframe(
              train_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=True,
              class_mode='categorical'
          )
   
  print("Shape of augmented training images:", train_generator.image_shape)
          
  valid_generator = valid_test_datagen.flow_from_dataframe(
              valid_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=True,
              class_mode='categorical'
          )
         
  print("Shape of validation images:", valid_generator.image_shape)
          
  test_generator = valid_test_datagen.flow_from_dataframe(
              test_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=False,
              class_mode='categorical'
          )
          
  print("Shape of test images:", test_generator.image_shape)
          
  return train_generator, valid_generator, test_generator


train_augmented, valid_augmented, test_augmented = augment_data(train_data, valid_data, test_data)


def show_images(gen):
      
    g_dict = gen.class_indices        
    classes = list(g_dict.keys())
    images, labels = next(gen)
    length = len(labels)        
    sample = min(length, 20)   
    plt.figure(figsize= (15, 17))
    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255      
        plt.imshow(image)
        index = np.argmax(labels[i])  
        class_name = classes[index]  
        plt.title(class_name, color= 'blue', fontsize= 7 )
        plt.axis('off')
    plt.show()



classes = len(list(train_augmented.class_indices.keys()))

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)

predictions = Dense(classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(
    train_augmented,
    epochs=1, ## sets the # of epochs, more epochs, better results.
    validation_data=valid_augmented,
)


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
print("Training Accuracy:", train_accuracy[-1])
print("Validation Accuracy:", val_accuracy[-1])
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def plot_actual_vs_predicted(model, test_data, num_samples=3):
    test_images, test_labels = next(iter(test_data))
    
    predictions = model.predict(test_images)
        
    class_labels = list(train_augmented.class_indices.keys())
        
    sample_indices = np.random.choice(range(len(test_images)), num_samples, replace=False)
    for i in sample_indices:
            actual_label = class_labels[np.argmax(test_labels[i])]
            predicted_label = class_labels[np.argmax(predictions[i])]
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(test_images[i].astype(np.uint8))  
            plt.title(f'Actual: {actual_label}')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(test_images[i].astype(np.uint8))  
            plt.title(f'Predicted: {predicted_label}')
            plt.axis('off')
            plt.show()
  

# print("HER IN NEW TESt")
# testDataDir="C:\\Users\\cimen\\Desktop\\code\\tenserflow\\testcontent"
# dataSplit = EyeDiseaseDataset(testDataDir)
# test_data = dataSplit.split_()
# test_augmented = augment_data(train_data, valid_data, test_data)
# print("NEW TEST FILEs")
# print(test_data)

plot_actual_vs_predicted(model, test_augmented)
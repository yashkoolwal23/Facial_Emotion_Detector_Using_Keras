import keras
from keras.preprocessing.image import ImageDataGenerator         #will do image augmentation to artificially expand the data-set
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
#Keras Conv2D is a 2D Convolution Layer, this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs.Kernel: In image processing kernel is a convolution matrix or masks which can be used for blurring, sharpening, embossing, edge detection, and more by doing a convolution between a kernel and an image.
#it creates many layers
from keras.layers import Conv2D , MaxPooling2D    #Maxpooling is used to take more impotant features by looking at feature map of conv2d diff layers        
import os

num_classes=5                                 #i.e only 5 in to train like angry,sad....
img_rows,img_cols = 48,48                     #image size we can say
batch_size = 8                                #at a particular time how many img you have to give model to train
train_data_dir = r'E:\Projects1\Facial Expression Recognition using Keras\train'                   #as python takes everything in forward slash now to give back slash we have used r
validation_data_dir = r'E:\Projects1\Facial Expression Recognition using Keras\validation'

#this will actaully create more images of the image by using following parameters 
train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=30 ,shear_range=0.3 , zoom_range=0.3, width_shift_range=0.4,
height_shift_range=0.4, horizontal_flip=True, vertical_flip=True )

#this will help to validate them
validation_datagen = ImageDataGenerator(rescale  = 1./255)

#in this now this we will give to model where flow from dir means from which dir and shuffle means that model needs to tained properly by looking at all images in batch size8 it will look happy as well as  sad.....
train_generator = train_datagen.flow_from_directory(train_data_dir,color_mode='grayscale',target_size=(img_rows,img_cols ),
batch_size=batch_size, class_mode='categorical',shuffle=True)


validation_generator = validation_datagen.flow_from_directory(validation_data_dir,color_mode='grayscale',target_size=(img_rows,img_cols ),
batch_size=batch_size, class_mode='categorical',shuffle=True)

# elu function i.e activation func it helps in setting the threshold value
# softmax function is to generate output b/w those 5 it actually helps in differentiating /// Softmax function, a wonderful activation function that turns numbers aka logits into probabilities that sum to one. Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes. 

model=Sequential()

#3x3 is just a matix that we want to take feature detector    //last 1 is for grayscale
#Block1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())                             #Batch normalization is used to make the model fast,stable
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())   
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))                    #to prevent overfitting

#Block2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())                             #Batch normalization is used to make the model fast,stable
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())   
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block3
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())                             #Batch normalization is used to make the model fast,stable
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())   
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block4
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())                             #Batch normalization is used to make the model fast,stable
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())   
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Block5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())   
model.add(Dropout(0.5))

#Block6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())   
model.add(Dropout(0.5))

#Block7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))


print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'E:\Projects1\Facial Expression Recognition using Keras\Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)



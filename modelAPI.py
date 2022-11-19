import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, concatenate

imH = 500
imW = 500

# ---------------------------------------------------------------------------------------------------------------

p0 = Input(shape=(imH,imW,3,), name='Plane0')

p0_conv1 = Conv2D(32, (11,11), activation = 'relu')(p0)
p0_pool1 = MaxPooling2D(pool_size = (2,2))(p0_conv1)

p0_conv2 = Conv2D(64, (3,3), activation = 'relu')(p0_pool1)
p0_pool2 = MaxPooling2D(pool_size = (3,3))(p0_conv2)

p0_tower1 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p0_pool2)

p0_tower2 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p0_pool2)
p0_tower2 = Conv2D(64, (3,3), padding='same', activation = 'relu')(p0_tower2)

p0_tower3 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p0_pool2)
p0_tower3 = Conv2D(64, (5,5), padding='same', activation = 'relu')(p0_tower3)

p0_tower4 = MaxPooling2D(pool_size = (3,3), strides=(1,1), padding='same')(p0_pool2)
p0_tower4 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p0_tower4)

p0_concat = concatenate([p0_tower1,p0_tower2,p0_tower3,p0_tower4], axis=1)

p0_pool3 = MaxPooling2D(pool_size = (3,3))(p0_concat)

# ---------------------------------------------------------------------------------------------------------------

p1 = Input(shape=(imH,imW,3,), name='Plane1')

p1_conv1 = Conv2D(32, (11,11), activation = 'relu')(p1)
p1_pool1 = MaxPooling2D(pool_size = (2,2))(p1_conv1)

p1_conv2 = Conv2D(64, (3,3), activation = 'relu')(p1_pool1)
p1_pool2 = MaxPooling2D(pool_size = (3,3))(p1_conv2)

p1_tower1 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p1_pool2)

p1_tower2 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p1_pool2)
p1_tower2 = Conv2D(64, (3,3), padding='same', activation = 'relu')(p1_tower2)

p1_tower3 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p1_pool2)
p1_tower3 = Conv2D(64, (5,5), padding='same', activation = 'relu')(p1_tower3)

p1_tower4 = MaxPooling2D(pool_size = (3,3), strides=(1,1), padding='same')(p1_pool2)
p1_tower4 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p1_tower4)

p1_concat = concatenate([p1_tower1,p1_tower2,p1_tower3,p1_tower4], axis=1)

p1_pool3 = MaxPooling2D(pool_size = (3,3))(p1_concat)

# ---------------------------------------------------------------------------------------------------------------

p2 = Input(shape=(imH,imW,3,), name='Plane2')

p2_conv1 = Conv2D(32, (11,11), activation = 'relu')(p2)
p2_pool1 = MaxPooling2D(pool_size = (2,2))(p2_conv1)

p2_conv2 = Conv2D(64, (3,3), activation = 'relu')(p2_pool1)
p2_pool2 = MaxPooling2D(pool_size = (3,3))(p2_conv2)

p2_tower1 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p2_pool2)

p2_tower2 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p2_pool2)
p2_tower2 = Conv2D(64, (3,3), padding='same', activation = 'relu')(p2_tower2)

p2_tower3 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p2_pool2)
p2_tower3 = Conv2D(64, (5,5), padding='same', activation = 'relu')(p2_tower3)

p2_tower4 = MaxPooling2D(pool_size = (3,3), strides=(1,1), padding='same')(p2_pool2)
p2_tower4 = Conv2D(64, (1,1), padding='same', activation = 'relu')(p2_tower4)

p2_concat = concatenate([p2_tower1,p2_tower2,p2_tower3,p2_tower4], axis=1)

p2_pool3 = MaxPooling2D(pool_size = (3,3))(p2_concat)

# ---------------------------------------------------------------------------------------------------------------

merge = keras.layers.concatenate([p0_pool3, p1_pool3, p2_pool3])

merge_tower1 = Conv2D(64, (1,1), padding='same', activation = 'relu')(merge)

merge_tower2 = Conv2D(64, (1,1), padding='same', activation = 'relu')(merge)
merge_tower2 = Conv2D(64, (3,3), padding='same', activation = 'relu')(merge_tower2)

merge_tower3 = Conv2D(64, (1,1), padding='same', activation = 'relu')(merge)
merge_tower3 = Conv2D(64, (5,5), padding='same', activation = 'relu')(merge_tower3)

merge_tower4 = MaxPooling2D(pool_size = (3,3), strides=(1,1), padding='same')(merge)
merge_tower4 = Conv2D(64, (1,1), padding='same', activation = 'relu')(merge_tower4)

merge_concat = concatenate([merge_tower1,merge_tower2,merge_tower3,merge_tower4], axis=1)

merge_pool = MaxPooling2D(pool_size = (3,3))(merge_concat)

flat = Flatten()(merge_pool)

den1 = Dense(units = 128, activation = 'relu')(flat)
drop1 = Dropout(rate=0.5,seed=7)(den1)
den2 = Dense(units = 64, activation = 'relu')(drop1)

predictions = Dense(units = 6, activation = 'softmax')(den2)

model = Model(inputs=[p0, p1, p2], outputs=predictions)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model.summary()

#from keras.utils import plot_model
#plot_model(model, to_file='model.pdf', show_shapes=False)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

def generate_generator_multiple(generator,dir1, dir2, dir3, batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)
    print genX1.class_indices
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)
    print genX2.class_indices
    
    genX3 = generator.flow_from_directory(dir3,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)

    print genX3.class_indices
    
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i[0], X3i[0]], X3i[1]  #Yield all images and their mutual label
            
training_set = generate_generator_multiple(generator = train_datagen,
                                           dir1 = '0dataset/training_set',
                                           dir2 = '1dataset/training_set',
                                           dir3 = '2dataset/training_set',
                                           batch_size = 8,
                                           img_height = imH,
                                           img_width = imW)

test_set = generate_generator_multiple(generator = test_datagen,
                                           dir1 = '0dataset/test_set',
                                           dir2 = '1dataset/test_set',
                                           dir3 = '2dataset/test_set',
                                           batch_size = 8,
                                           img_height = imH,
                                           img_width = imW)

model.fit_generator(training_set,
                    steps_per_epoch = 1250,
                    epochs = 1,
                    validation_data = test_set,
                    validation_steps = 312,
                    use_multiprocessing=True,
                    shuffle=False)




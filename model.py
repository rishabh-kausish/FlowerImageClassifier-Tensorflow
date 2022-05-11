import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from workspace_utils import active_session
import json


batch_size = 64
image_size = 224

def norm(image,label):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image ,(image_size,image_size))
    image/=255
    return image,label


if __name__ == '__main__':
    try:
        (train_set,test_set,validation_set),ds_info = tfds.load('oxford_flowers102',as_supervised = True,with_info = True,split =           ['train','test','validation'])
        print('Successfully loaded the data')
    except Exception as e:
        print(f'Error while loading the data : {e}')
    train_examples = ds_info.splits['train'].num_examples
    test_examples = ds_info.splits['test'].num_examples
    train_examples = ds_info.splits['train'].num_examples
    val_examples = ds_info.splits['validation'].num_examples
    num_labels = ds_info.features['label'].num_classes
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
        
    # initializing batches
    
    train_batches = train_set.shuffle(train_examples).map(norm).batch(batch_size).prefetch(1)
    test_batches = test_set.map(norm).batch(batch_size).prefetch(1)
    validation_batches = validation_set.map(norm).batch(batch_size).prefetch(1)
    
    #training the model
    with active_session():
        URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

        feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
        feature_extractor.trainable = False

        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(102,activation = 'softmax')

        ])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',patience = 5)
        model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
        history = model.fit(train_batches,epochs = 12,callbacks = [early_stopping],validation_data = validation_batches,batch_size = batch_size)
    
    #saving the model
    model.save('my_model.h5')

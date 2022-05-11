import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import tensorflow_hub as hub
from workspace_utils import active_session
import json
'''

If getting error while executing this file:
Please update the following packages-

pip --no-cache-dir install tensorflow-datasets --user
pip --no-cache-dir install tfds-nightly --user
pip --no-cache-dir install --upgrade tensorflow --user
'''


parser = argparse.ArgumentParser()
parser.add_argument('img_path', help = 'Path of the image path expected')
parser.add_argument('model', help = 'Path of model file')
parser.add_argument(
   '--top_k', type = int, nargs = 1)
parser.add_argument(
   '--category_names', nargs = 1)
args = parser.parse_args()

if args.category_names:
    category = 1
    with open(args.category_names[0],'r') as f:
        class_names = json.load(f)
    print(class_names)
else:
    category = 0
    
if args.top_k:
    top_k = args.top_k[0]
else:
    top_k = 1

def process_image(img):
    img = np.asarray(img)
    img = tf.cast(img,tf.float32)
    img = tf.image.resize(img,size = (224,224))
    img/=255
    return img.numpy()

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = np.expand_dims(img,axis = 0)
    preds = model.predict(img)
    indexes = np.argsort(preds.squeeze())[::-1][0:top_k]
    probs = list()
    classes = list()
    for i in indexes:
        probs.append(preds.squeeze()[i])
        if category == 1:
            classes.append(class_names[str(i+1)])
        else:
            classes.append(i+1)
        
    return probs,classes



        
if __name__ =='__main__':
    
    with active_session():
        try:
            saved_model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer': hub.KerasLayer})
            print('Successfully loaded the model')
        except Exception as e:
            print(f'Error while loading the model : {e}')
        
        probs,classes = predict(args.img_path, saved_model,top_k)
        print(classes)
        
    
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np

from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout


from tqdm import tqdm
import pandas as pd
#tqdm().pandas()



def load_doc(filename):

    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    # print(len(captions))
    descriptions ={}
    for caption in captions[:-1]:

        val = tuple(caption.split(','))
        if len(val)!=2:
            continue
        img, caption = caption.split(',')
        if img not in descriptions:
            descriptions[img] = [caption]
        else:
            descriptions[img].append(caption)
    return descriptions



def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):

            img_caption.replace("-"," ")
            desc = img_caption.split()


            desc = [word.lower() for word in desc]

            desc = [word.translate(table) for word in desc]

            desc = [word for word in desc if(len(word)>1)]

            desc = [word for word in desc if(word.isalpha())]


            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions


def text_vocabulary(descriptions):

    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()


dataset_text = "/home/balaji/Documents/MiniProject/Dataset/archive"
dataset_images = "/home/balaji/Documents/MiniProject/Dataset/archive/Images"


filename = dataset_text + "/" + "captions.txt"

descriptions = all_img_captions(filename)
print("Length of descriptions =" ,len(descriptions))


clean_descriptions = cleaning_text(descriptions)


vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))


save_descriptions(clean_descriptions, "descriptions.txt")


def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0

            feature = model.predict(image)
            features[img] = feature
        return features

features = extract_features(dataset_images)



dump(features, open("features.p","wb"))

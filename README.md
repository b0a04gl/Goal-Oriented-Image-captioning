# Goal-Oriented-Image-captioning

Involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.


Dataset:

Link : https://vizwiz.org/tasks-and-datasets/image-captioning/

Name : VizWiz-Captions dataset
      23,431 training images
      117,155 training captions
      7,750 validation images
      38,750 validation captions
      8,000 test images
      40,000 test captions

Image files format: 
  images = [image]
  image = {
    "file_name": "VizWiz_train_00023410.jpg",
    "id": 23410
    "text_detected": true
  }

Captions file format:
  annotations = [annotation]
  annotation = {
      "image_id": 23410,
      "id": 117050,
      "caption": "A plastic rewards card lying face down on the floor."
      "is_rejected": false,
      "is_precanned": false,
      "text_detected": true
  }


Requirements:

1.Tensorflow
2.Keras
3.Numpy
4.h5py
5.Pandas  
6.Pillow

Flow : 
  1. Getting and performing data cleaning
  2. Extracting the feature vector from all images 
  3. Loading dataset for Training the model
  4. Tokenizing the vocabulary 
  5. Defining the CNN-RNN model
  6. Train the model
  7. Test the model
  

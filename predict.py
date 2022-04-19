import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow import keras
from PIL import Image

# load actual weights
model = keras.models.load_model('best_model10000.h5')
# load the tokenizer
tokenizer = pickle.load(open('tokenizer10000.pkl', 'rb'))
# load vgg16 model
modelvgg16 = VGG16()
# restructure the model
modelvgg16 = Model(inputs=modelvgg16.inputs, outputs=modelvgg16.layers[-2].output)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    # img_path = os.path.join(BASE_DIR, "Images", image_name)
    img_path = image_name
    image = Image.open(img_path)
    # captions = mapping[image_id]
    # print('---------------------Actual---------------------')
    # for caption in captions:
    #     print(caption)
    # predict the caption

    # load the image from file
    img_path = image_name
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = modelvgg16.predict(image, verbose=0)
    # get image ID
    # image_id = img_name.split('.')[0]
    # store feature
    # features[image_id] = feature

    y_pred = predict_caption(model, feature, tokenizer, max_length = 13)
    # print('--------------------Predicted--------------------')
    # print(y_pred)
    return " ".join(y_pred.split(" ")[1:-1])
    # plt.imshow(image)

# now can use generate_caption to predict data
generate_caption("00064.jpg")


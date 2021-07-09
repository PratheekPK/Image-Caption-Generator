import string
import numpy as np
from PIL import Image
from pickle import dump
import os
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tqdm import tqdm_notebook as tqdm

tqdm().pandas()

text_tokens = (
    r"C:\Users\hp\Code\GitHub\Image Caption Generator\Flickr8k_text\Flickr8k.token.txt"
)

images = r"C:\Users\hp\Code\GitHub\Image Caption Generator\Flickr8k_Dataset\Flicker8k_Dataset"

file = open(text_tokens, "r")
whole_text = file.read()
file.close()
new_whole_text = whole_text.split("\n")
descriptions = {}

for line in new_whole_text[:-1]:
    image, line = line.split("\t")
    if image[:-2] not in descriptions:
        descriptions[image[:-2]] = [line]
    else:
        descriptions[image[:-2]].append(line)

table = str.maketrans("", "", string.punctuation)

for image, lines in descriptions.items():
    for index, line in enumerate(lines):
        line.replace("-", " ")
        desc = line.split()
        desc = [word.lower() for word in desc]
        desc = [word.translate(table) for word in desc]
        desc = [word for word in desc if (len(word) > 1)]
        desc = [word for word in desc if (word.isalpha())]
        line = " ".join(desc)
        descriptions[image][index] = line

vocab = set()
for key in descriptions.keys():
    [vocab.update(d.split()) for d in descriptions[key]]

lines = list()
for key, descript_list in descriptions.items():
    for desc in descript_list:
        lines.append(key + "\t" + desc)
de_script = "\n".join(lines)

model = Xception(include_top=False, pooling="avg")
features = {}
for img in tqdm(os.listdir(images)):
    file_name = images + "/" + img
    image = Image.open(file_name)
    image = image.resize((299, 299))
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    features[img] = feature

filename = r"C:\Users\hp\Code\GitHub\Image Caption Generator\Flickr8k_text\Flickr_8k.trainImages.txt"

file = open(filename, "r")
pics = file.read()
file.close()

pics = pics.split("\n")[:-1]
descriptions = {}

for line in de_script.split("\n"):
    words = line.split()
    if len(words) < 1:
        continue
    image, image_caption = words[0], words[1:]
    if image in pics:
        if image not in descriptions:
            descriptions[image] = []
        desc = "<start> " + " ".join(image_caption) + " <end>"
        descriptions[image].append(desc)

file = open("descriptions.txt", "w")
file.write(descriptions)
file.close()

features = {k: features[k] for k in pics}

all_desc = []
for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_desc)

dump(tokenizer, open("tokenizer.p", "wb"))
vocab_size = len(tokenizer.word_index) + 1

all_desc = []
for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]

max_length = max(len(d.split()) for d in descript_list)


def data_gen(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(
                tokenizer, max_length, description_list, feature
            )
            yield [[input_image, input_sequence], output_word]


def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam")

epochs = 10
steps = len(descriptions)

for i in range(epochs):
    generator = data_gen(descriptions, features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")

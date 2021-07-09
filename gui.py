import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
from pickle import load
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences

max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
xception_model = Xception(include_top=False, pooling="avg")
model = load_model("my_model.h5")


top = tk.Tk()
top.geometry("800x600")
top.title("Image Caption Generator")
top.configure(background="#ffffff")

label = Label(top, background="#ffffff", font=("arial", 20, "bold"))
sign_image = Label(top)


def word_for_id(pred, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == pred:
            return word
    return None


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    photo = xception_model.predict(image)

    in_text = ""
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)

        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break
    in_text = in_text.rsplit(" ", 1)[0]
    label.configure(foreground="#011638", text=in_text)


def show_classify_button(file_path):
    classify_b = Button(
        top,
        text="Generate Caption",
        command=lambda: classify(file_path),
        padx=10,
        pady=5,
    )
    classify_b.configure(
        background="#1500ff", foreground="white", font=("arial", 10, "bold")
    )
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text="")
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload the image", command=upload_image, padx=10, pady=5)
upload.configure(background="#1500ff", foreground="white", font=("arial", 14, "bold"))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(
    top, text="Image Caption Generator", pady=20, font=("arial", 20, "bold")
)
heading.configure(background="#ffffff", foreground="#1500ff")
heading.pack()
top.mainloop()

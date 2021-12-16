import re
import pickle
import numpy as np
import streamlit as st

import io
from PIL import Image

from model import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def load_model():
    # Dimension for the image embeddings and token embeddings
    EMBED_DIM = 512
    # Number of self-attention heads
    NUM_HEADS = 4
    # Per-layer units in the feed-forward network
    FF_DIM = 512

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS
    )

    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )
    # Define the loss function
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    # Compile the model
    caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)
    caption_model.load_weights("model_efficient4/modelEfficient")

    return caption_model

def custom_standardization(input_string):
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def word_map():
    # Vocabulary size
    VOCAB_SIZE = 10000
    # Fixed length allowed for any sequence
    SEQ_LENGTH = 20

    text_data_path = "textDataEfficient4.pickle"
    with open(text_data_path, "rb") as f:
        text_data = pickle.load(f)

    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    vocab_path = "vocabEfficient4.pickle"
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    return vectorization, index_lookup, max_decoded_sentence_length

def generate_caption(image, caption_model):
    vectorization, index_lookup, max_decoded_sentence_length = word_map()

    # Read the image from the disk
    img = read_image(image)
    img = img.numpy().astype(np.uint8)

     # Pass the image to the CNN
    img = tf.expand_dims(img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    predicted = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()

    return predicted

def read_image(img_path, size=(299,299)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def upload_image():
    args = { 'sunset' : 'sunset.jpeg' }
    
    img_upload  = st.file_uploader(label= 'Upload Image', type = ['png', 'jpg', 'jpeg','webp'])
    img_open = args['sunset'] if img_upload is None else img_upload

    image1 = Image.open(img_open)
    rgb_im = image1.convert('RGB')
    rgb_im.save("saved_image.jpeg")

    st.image(image1,use_column_width=True,caption="Your image")


if __name__ == '__main__':
    caption_model = load_model()

    st.title("The Image Captioning Bot")
    st.text("")
    st.text("")
    st.success("Welcome! Please upload an image!"
    )   

    upload_image()

    if st.button('Generate captions!'):
        predicted = generate_caption("saved_image.jpeg", caption_model)

        styled = f'''<p style="font-size: 20px;"><b>Predicted:</b></p>
        <p style="font-size: 20px;">{predicted}</p>'''
        st.markdown(styled, unsafe_allow_html=True)

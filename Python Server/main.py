from flask import Flask, render_template, url_for, request,jsonify
app = Flask(__name__)
from tkinter import *
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from keras import models
from keras.layers import Input,LSTM,Dense
from flask_cors import CORS
import json

CORS(app,origins=['http://localhost:3000'])

cv_translation=CountVectorizer(binary=True,tokenizer=lambda txt: txt.split(),stop_words=None,analyzer='char')

cv_transliteration=CountVectorizer(binary=True,tokenizer=lambda txt: txt.split(),stop_words=None,analyzer='char')

datafile_translation = pickle.load(open("training_data_translation.pkl","rb"))
input_characters_translation = datafile_translation['input_characters']
target_characters_translation = datafile_translation['target_characters']
max_input_length_translation = datafile_translation['max_input_length']
max_target_length_translation = datafile_translation['max_target_length']
num_en_chars_translation = datafile_translation['num_en_chars']
num_dec_chars_translation = datafile_translation['num_dec_chars']
input_texts_translation=datafile_translation['input_texts']

datafile_transliteration = pickle.load(open("training_data_transliteration.pkl","rb"))
input_characters_transliteration = datafile_transliteration['input_characters']
target_characters_transliteration = datafile_transliteration['target_characters']
max_input_length_transliteration = datafile_transliteration['max_input_length']
max_target_length_transliteration = datafile_transliteration['max_target_length']
num_en_chars_transliteration = datafile_transliteration['num_en_chars']
num_dec_chars_transliteration = datafile_transliteration['num_dec_chars']

@app.route('/')
def index():
  return render_template('index.html')
#Inference model
#load the model
model_transliteration = models.load_model("s2s_transliteration")
#construct encoder model from the output of second layer
#discard the encoder output and store only states.
enc_outputs_transliteration, state_h_enc_transliteration, state_c_enc_transliteration = model_transliteration.layers[2].output
#add input object and state from the layer.
en_model_transliteration = Model(model_transliteration.input[0], [state_h_enc_transliteration, state_c_enc_transliteration])
#create Input object for hidden and cell state for decoder
#shape of layer with hidden or latent dimension
dec_state_input_h_transliteration = Input(shape=(256,), name="input_6")
dec_state_input_c_transliteration = Input(shape=(256,), name="input_7")
dec_states_inputs_transliteration = [dec_state_input_h_transliteration, dec_state_input_c_transliteration]
#add input from the encoder output and initialize with states.
dec_lstm_transliteration = model_transliteration.layers[3]
dec_outputs_transliteration, state_h_dec_transliteration, state_c_dec_transliteration = dec_lstm_transliteration(
    model_transliteration.input[1], initial_state=dec_states_inputs_transliteration
)
dec_states_transliteration = [state_h_dec_transliteration, state_c_dec_transliteration]
dec_dense_transliteration = model_transliteration.layers[4]
dec_outputs_transliteration = dec_dense_transliteration(dec_outputs_transliteration)
#create Model with the input of decoder state input and encoder input
#and decoder output with the decoder states.
dec_model_transliteration = Model(
    [model_transliteration.input[1]] + dec_states_inputs_transliteration, [dec_outputs_transliteration] + dec_states_transliteration
)


def decode_sequence_transliteration(input_seq):
  # create a dictionary with a key as index and value as characters.
  reverse_target_char_index_transliteration = dict(enumerate(target_characters_transliteration))
  # get the states from the user input sequence
  states_value_transliteration = en_model_transliteration.predict(input_seq)

  # fit target characters and
  # initialize every first character to be 1 which is '\t'.
  # Generate empty target sequence of length 1.
  co = cv_transliteration.fit(target_characters_transliteration)
  target_seq_transliteration = np.array([co.transform(list("\t")).toarray().tolist()], dtype="float32")

  # if the iteration reaches the end of text than it will be stop the it
  stop_condition = False
  # append every predicted character in decoded sentence
  decoded_sentence = ""

  while not stop_condition:
    # get predicted output and discard hidden and cell state.
    output_chars, h, c = dec_model_transliteration.predict([target_seq_transliteration] + states_value_transliteration)

    # get the index and from the dictionary get the character.
    char_index = np.argmax(output_chars[0, -1, :])
    text_char = reverse_target_char_index_transliteration[char_index]
    decoded_sentence += text_char
    # Exit condition: either hit max length
    # or find a stop character.
    if text_char == "\n" or len(decoded_sentence) > max_target_length_transliteration:
      stop_condition = True
    # update target sequence to the current character index.
    target_seq_transliteration = np.zeros((1, 1, num_dec_chars_transliteration))
    target_seq_transliteration[0, 0, char_index] = 1.0
    states_value_transliteration = [h, c]
  # return the decoded sentence
  return decoded_sentence


def bagofcharacter_transliteration(input_t):
  cv_transliteration = CountVectorizer(binary=True, tokenizer=lambda txt:
  txt.split(), stop_words=None, analyzer='char')
  en_in_data = [];
  pad_en = [1] + [0] * (len(input_characters_transliteration) - 1)

  cv_inp = cv_transliteration.fit(input_characters_transliteration)
  en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

  if len(input_t) < max_input_length_transliteration:
    for _ in range(max_input_length_transliteration - len(input_t)):
      en_in_data[0].append(pad_en)

  return np.array(en_in_data, dtype="float32")

#Inference model
#load the model
model_translation = models.load_model("model_translation")
#construct encoder model from the output of second layer
#discard the encoder output and store only states.
enc_outputs_translation, state_h_enc_translation, state_c_enc_translation = model_translation.layers[2].output
#add input object and state from the layer.
en_model_translation = Model(model_translation.input[0], [state_h_enc_translation, state_c_enc_translation])
#create Input object for hidden and cell state for decoder
#shape of layer with hidden or latent dimension
dec_state_input_h_translation = Input(shape=(256,))
dec_state_input_c_translation = Input(shape=(256,))
dec_states_inputs_translation = [dec_state_input_h_translation, dec_state_input_c_translation]
#add input from the encoder output and initialize with states.
dec_lstm_translation = model_translation.layers[3]
dec_outputs_translation, state_h_dec_translation, state_c_dec_translation = dec_lstm_translation(
    model_translation.input[1], initial_state=dec_states_inputs_translation
)
dec_states_translation = [state_h_dec_translation, state_c_dec_translation]
dec_dense_translation = model_translation.layers[4]
dec_outputs_translation = dec_dense_translation(dec_outputs_translation)
#create Model with the input of decoder state input and encoder input
#and decoder output with the decoder states.
dec_model_translation = Model(
    [model_translation.input[1]] + dec_states_inputs_translation, [dec_outputs_translation] + dec_states_translation
)


def decode_sequence_translation(input_seq):
  # create a dictionary with a key as index and value as characters.
  reverse_target_char_index_translation = dict(enumerate(target_characters_translation))
  # get the states from the user input sequence
  states_value_translation = en_model_translation.predict(input_seq)

  # fit target characters and
  # initialize every first character to be 1 which is '\t'.
  # Generate empty target sequence of length 1.
  co_translation = cv_translation.fit(target_characters_translation)
  target_seq_translation = np.array([co_translation.transform(list("\t")).toarray().tolist()], dtype="float32")

  # if the iteration reaches the end of text than it will be stop the it
  stop_condition = False
  # append every predicted character in decoded sentence
  decoded_sentence_translation = ""

  while not stop_condition:
    # get predicted output and discard hidden and cell state.
    output_chars_translation, h_translation, c_translation = dec_model_translation.predict(
      [target_seq_translation] + states_value_translation)

    # get the index and from the dictionary get the character.
    char_index_translation = np.argmax(output_chars_translation[0, -1, :])
    text_char_translation = reverse_target_char_index_translation[char_index_translation]
    decoded_sentence_translation += text_char_translation
    # Exit condition: either hit max length
    # or find a stop character.
    if text_char_translation == "\n" or len(decoded_sentence_translation) > max_target_length_translation:
      stop_condition = True
    # update target sequence to the current character index.
    target_seq_translation = np.zeros((1, 1, num_dec_chars_translation))
    target_seq_translation[0, 0, char_index_translation] = 1.0
    states_value_translation = [h_translation, c_translation]
  # return the decoded sentence
  return decoded_sentence_translation


def bagofcharacter_translation(input_t):
  cv_translation = CountVectorizer(binary=True, tokenizer=lambda txt:
  txt.split(), stop_words=None, analyzer='char')
  en_in_data = [];
  pad_en = [1] + [0] * (len(input_characters_translation) - 1)

  cv_inp_translation = cv_translation.fit(input_characters_translation)
  en_in_data.append(cv_inp_translation.transform(list(input_t)).toarray().tolist())

  if len(input_t) < max_input_length_translation:
    for _ in range(max_input_length_translation - len(input_t)):
      en_in_data[0].append(pad_en)

  return np.array(en_in_data, dtype="float32")

@app.route('/translate_to_Konkani', methods=['POST', 'GET'])
def translate_to_Konkani():
  if request.method == "POST":
    sent=request.get_json()
    input_text = sent[0]['text'].split(',')
    print(input_text)
    print(type(input_text))
    output_texts = ""
    for x in input_text:
      print(x)
      term = x + "."
      if term in input_texts_translation:
        en_in_data = bagofcharacter_translation(x.lower() + ".")
        x = decode_sequence_translation(en_in_data)
        output_texts += " " + x
      else:
        en_in_data = bagofcharacter_transliteration(x.lower() + ".")
        x = decode_sequence_transliteration(en_in_data)
        output_texts += " " + x
    o={'kon':output_texts}
    return jsonify(o)

if __name__ == "__main__":
  app.run(debug=True)
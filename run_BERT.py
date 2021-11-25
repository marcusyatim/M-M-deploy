'''
Author: Eu Jin Marcus Yatim
This python script takes in the review and runs it through the fine-tuned DistilBERT base model from Hugging Face 🤗. The results from the model is then post-processed to get the rating number.
Requires fine-tuned model weights from /data/ratings-classification/.
Use fine_tune_BERT.ipynb to create the required data.
'''
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress warnings
import tensorflow as tf
import transformers

from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

transformers.logging.set_verbosity_error()  # Suppress warnings
checkpoint = 'distilbert-base-uncased'

def load_bert():
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5,
        end_learning_rate=0.,
        decay_steps=4020
        )
    opt = Adam(learning_rate=lr_scheduler)
    loss = CategoricalCrossentropy(from_logits=True)

    # Load model
    model = TFAutoModelForSequenceClassification.from_pretrained("/app/data/getRatings/", num_labels=5, problem_type="multi_label_classification")
    model.compile(optimizer=opt, loss=loss)

    return model

def convert_ratings(predictions):
    # Find the max probability value and extract the index of that value. The index corresponds to the rating value (1-5). As such, add '1' to get actual rating value.
    rating = tf.math.argmax(predictions, 1).numpy()[0] + 1

    return rating

def get_results(review):
    # Initialise Huggingface tonenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenize the review
    tonenized_review = tokenizer(review, truncation=True, padding=True, return_tensors='tf')

    # Load the model
    model = load_bert()

    # Input the tonenized review into the model and receive the logits from the results of the model
    output = model(tonenized_review)

    # Pass the logits through softmax layer to get probability values
    predictions = tf.math.softmax(output.logits, axis=-1)

    # Get the rating from the probability values
    rating = convert_ratings(predictions)

    return rating

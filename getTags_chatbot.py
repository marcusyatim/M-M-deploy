'''
Author: Eu Jin Marcus Yatim
This python script asks the user to input a foodnetwork.com recipe url, performs web scrapping on it and inputs the data into the transformer to return tags for the recipe.
This script is to run on the chatbot.
Requires run_transformer.py
'''
import requests
import sys
import run_transformer

from bs4 import BeautifulSoup

def webscrapping(url):
    '''
    This function performs web scraping on a foodnetwork.com recipe.
    Returns the recipe information as a list.
    '''
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all('li', attrs={'class': 'o-Method__m-Step'})
    recipe_info = []
    for result in results:
      recipe_info.append(result.text.strip())

    return recipe_info

def preprocess(sentence):
    '''
    This function takes in a string and removes certain punctuations from it.
    It will also append the start and end token to the string,
    Returns the preprocessed string.
    '''

    # Strip "[]'," from the sentence
    sentence = sentence.translate(str.maketrans('', '', ".[]',"))

    # Adding a start and an end token to the sentence so that the model know when to start and stop predicting.
    sentence = '<start> ' + sentence + ' <end>'

    return sentence

######################
####### MAIN #########
######################
def getTags(url):
    # Perform web scrapping of the url
    recipe_info = webscrapping(url)

    # Preprocess the target recipe info
    preprocessed_recipe = preprocess(str(recipe_info))

    # Get tags
    tags = run_transformer.getTags(preprocessed_recipe)

    return tags

if __name__ == '__main__':
    main()
'''
Author: Eu Jin Marcus Yatim
This python script asks the user to input their likes and dislikes and matches those to the closest recipes.
This script is to run on the chatbot.
Requires NLTK packages: stopwords, punkt and wordnet. Run setup.py to install them.
Requires an 'assignment' CSV file and a 'topN' JSON file.
'''
import json
import nltk
import pandas as pd

# from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

def preprocess(text):
    '''
    This function takes in a string and tokenises each string in the list, as well as filtering out unwated tokens.
    The tokens are then returned
    '''
    # mystopwords = stopwords.words("english")
    WNlemma = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [ t for t in tokens if t.isalpha() ]
    tokens = [ WNlemma.lemmatize(t.lower()) for t in tokens ]
    # tokens = [ t for t in tokens if t not in mystopwords ]
    tokens = [ t for t in tokens if len(t) >= 3 ]

    return tokens

def wordnet(list):
    '''
    This function takes in a list of tokens and gets the synonyms, hyponyms, hypernyms, meronyms, holonyms & entailments for each token.
    Duplicates are avoided by using set().
    Returns them as a set.
    '''
    wordnet = set()
    for token in list:
        for synset in wn.synsets(token):
            for lemma in synset.lemmas():
                wordnet.add(lemma.name())
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemma_names():
                    wordnet.add(lemma)
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemma_names():
                    wordnet.add(lemma)
            for meronym in synset.part_meronyms():
                for lemma in meronym.lemma_names():
                    wordnet.add(lemma)
            for holonym in synset.part_holonyms():
                for lemma in holonym.lemma_names():
                    wordnet.add(lemma)
            for entailment in synset.entailments():
                for lemma in entailment.lemma_names():
                    wordnet.add(lemma)

    return wordnet 

def match_topics(likes, dislikes, topic_dict):
    '''
    This function uses a simple score based system. 
    If the word in likes is found in a topic, score +1 for that topic.
    If the word in dislikes is found in a topic, score -1 for that topic.
    Scores are saved in a dictionary with the topics as keys and scores as values. If score for a topic is <1, do not consider that topic.
    If no topics have score >0, return error message and exit. Else return a sorted top 3 highest scoring topics as a list.
    '''
    score_dict = {}
    for key, value in topic_dict.items():
        score = 0
        for like in likes:
            if like in value:
                score += 1
        for dislike in dislikes:
            if dislike in value:
                score -= 1
        if score < 1:
            continue
        else:
            score_dict[key] = score
    if not score_dict:
        output = "Your search did not produce any results. Kindly try again with a more specific search."
        return output
    else:
        likely_topics = sorted(score_dict, key=score_dict.get, reverse=True)[:3]

    return likely_topics

def match_recipes(exp, topics):
    '''
    This function sorts the recipes based on the likely topics (up to 3) and with priority based on the topics ranked order.
    After sorting, extract the ids of the top 5 recipes. Returns a list of these recipe ids.
    '''
    likely_recipes = []
    recipes = pd.read_csv('/app/data/getRecommendations/assignments/' + exp + '.csv', usecols=['id'] + topics)

    # This line shuffles the dataframe. This is to create randomness as many recipes have the same probabilities.
    recipes = recipes.sample(frac=1)

    if len(topics) == 1:
        recipes = recipes.sort_values(topics[0], ascending=False)
    elif len(topics) == 2:
        recipes = recipes.sort_values([topics[0], topics[1]], ascending=False)
    else:
        recipes = recipes.sort_values([topics[0], topics[1], topics[2]], ascending=False)
    recipes = recipes['id'].head(3)
    for recipe in recipes:
        likely_recipes.append(recipe)
    
    return likely_recipes

def recipe_links(recipes):
    '''
    This function prints out the Food.com links of the likely recipes, based on their ids.
    '''
    output = "Your Top 3 recommended recipes are:"
    for recipe in recipes:
        output += "\nhttps://www.food.com/recipe/" + str(recipe)

    return output

######################
####### MAIN #########
######################
def getRecommendations(likes, dislikes):
    # Define experiment to be used
    exp = 'exp3'

    # Preprocess the likes and dislikes (remove stopwords, punctuations, tokenisation)
    likes_tokens = preprocess(likes)
    dislikes_tokens = preprocess(dislikes)

    # Get wordnet of likes and dislikes
    likes_wordnet = wordnet(likes_tokens)
    dislikes_wordnet = wordnet(dislikes_tokens)

    # Get topic dictionary
    topic_dict = json.load(open('/app/data/getRecommendations/topN/' + exp + '.json'))

    # Match likes and dislikes to the most likely topic(s)
    likely_topics = match_topics(likes_wordnet, dislikes_wordnet, topic_dict)
    if type(likely_topics) == str:
        return likely_topics

    # Match topic with recipes
    likely_recipes = match_recipes(exp, likely_topics)    

    # Output the recipe link to Food.com
    links = recipe_links(likely_recipes)

    return links
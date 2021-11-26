'''
Author: Eu Jin Marcus Yatim
This python script asks the user to input a review for a recipe. The script will then automatically give a rating (from 1-5) of the recipe based on the review.
This script is to run on the chatbot.
Requires run_BERT.py
'''
import sys
import run_BERT

######################
####### MAIN #########
######################
def getRatings(review):
    # Run BERT to get the rating of the review
    rating = run_BERT.get_results(review)

    return str(rating)

if __name__ == '__main__':
    main()
import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    # Get the test sequences from test_set.get_all_Xlengths().values()
    # Starter Code format used from Udacity Discussion Forums
    test_sequences = list(test_set.get_all_Xlengths().values())
    for test_X,test_length in test_sequences:
        prob_word = {}
        # Go through individual words and get their scores
        # In case of error set proability to float value of -inf
        for word, model in models.items():
            try:
                logL = model.score(test_X,test_length)
                prob_word[word] = logL
            except:
                prob_word[word] = float("-inf")
        # Append Individual Probabilities to Total Proability
        probabilities.append(prob_word)
        # Take the Max Guess Word from the individual Word Probability Dictionary
        guess_word = max([(v,k) for (k,v) in prob_word.items()])[1]
        # Append the same to total guesses
        guesses.append(guess_word)
    return (probabilities, guesses)
    raise NotImplementedError

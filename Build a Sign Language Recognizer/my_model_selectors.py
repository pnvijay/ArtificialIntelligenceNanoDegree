import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    ## create a function wherein BIC Score can be calculated
    def get_BIC(self,n):
        # Define the model
        model = GaussianHMM(n_components = n, covariance_type='diag',n_iter=1000,random_state = self.random_state,
                                    verbose=False).fit(self.X,self.lengths)
        # Calculate the model score and then the BIC Score based on formula
        logL = model.score(self.X,self.lengths)
        n_param = n * n + 2 * n * len(self.X[0]) - 1
        BIC_score = -2 * logL + n_param + np.log(len(self.X))
        return BIC_score
    
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        # Define best score and best_n
        best_score = float('inf')
        best_n = None
        # Loop Through the components and get the best score by comparing with BIC score for the component
        # Update the best score and best_n
        # Return Model with the best_n
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                BIC_score = self.get_BIC(n_components)
                if  best_score > BIC_score:
                    best_score = BIC_score
                    best_n = n_components
            except:
                pass
        return self.base_model(best_n)
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    ## create a function wherein DIC can be calculated
    def get_DIC(self,n):
        # Define a generic Model
        model = self.base_model(n)
        # Calculate score on the model for the word
        score_this_word = model.score(self.X,self.lengths)
        score_rest = 0
        # Form a Corpus of words that are different from the given word
        rest_of_words = [word for word in self.words if word != self.this_word]
        V = len(rest_of_words)
        # Calculate the model score for all those words and then sum them up
        for w in rest_of_words:
            d, e = self.hwords[w]
            score_word = model.score(d,e)
            score_rest += score_word
        # Calculate the DIC score based on the formula
        DIC_score = score_this_word - score_rest / (V-1)
        return DIC_score

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        # Define best score and best_n
        best_score = float('-inf')
        best_n = None
        # Loop Through the components and get the best score by comparing with DIC score for the component
        # Update the best score and best_n
        # Return Model with the best_n
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                DIC_score = self.get_DIC(n_components)
                if  best_score < DIC_score:
                    best_score = DIC_score
                    best_n = n_components
            except:
                pass
        return self.base_model(best_n)
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        # Define best score and best_n
        best_score = float('-inf')
        best_n = None
        best_model = None
        # Loop Through the components and get the best score by comparing with BIC score for the component
        # Update the best score and best_n
        # Return Model with the best_n
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                split_method = KFold(n_splits=min(3,len(self.lengths))) #This definition of split method is as per the suggestion by Udacity Mentors in Discussion Forums
                logL = []  # list of cross validation scores obtained
                # get the model for the combined cross-validation training sequences and score with their combined
                #  validation sequences filling the list 'logL'
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train, lengths_train = combine_sequences(cv_train_idx,self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx,self.sequences)
                    model = GaussianHMM(n_components, covariance_type='diag',n_iter=1000,random_state = self.random_state,
                                        verbose=False).fit(X_train,lengths_train)
                    foldL = model.score(X_test,lengths_test)
                    logL.append(foldL)
                # Get the average of those fold scores
                mean_score = np.mean(logL)
                # if the averave score is better than the best score then update the best score and choose model with best_n after fitting with all of X.
                if  mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_n = n_components
                ##best_model = GaussianHMM(best_n, covariance_type='diag',n_iter=1000,random_state = self.random_state,
                                        ##verbose=False).fit(self.X,self.lengths)
                    best_model = self.base_model(best_n)
            except:
                pass
        return best_model
        raise NotImplementedError

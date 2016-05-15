###############################################################################
# ------------------------ Imports and Global Vars -------------------------- #
###############################################################################

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from numpy import mean
from scipy.stats import norm
from textblob import TextBlob
from textblob.exceptions import TranslatorError, NotTranslated
from urllib2 import HTTPError

# initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# initialize normal distribution
norm_dist = norm(0, 30)
dist_normal_value = norm_dist.pdf(0)


###############################################################################
# --------------------------- Main Functionality ---------------------------- #
###############################################################################

def evaluate_chat(messages, rank, alpha=0.6, translate_input=False):
    """
    evaluates a chat sentimental value by the following formula:
    chat_value = max(mean(messages_neg_sentiments), mean(messages_pos_sentiments)) * alpha
                 + Norm(0,30)[rank] * (1 - alpha)

    meaning we take the mean of the positive and negative sentiments of the conversation, choose the maximum of both
    and evaluating it while taking into consideration how popular it ranks among other conversations under a normal
    distribution.
                                                        2
                                                      -x
       /  n          pos    n          neg\          ----
       |===== message     ===== message   |          1800
       |\            i    \            i  |         e
    max| >    ----------,  >    ----------| alpha + ----- (1 - alpha)
       |/          n      /          n    |            __
       |=====             =====           |         60 ||
       \i = 1             i = 1           /

    :param messages: the chat messages (list of strings)
    :param rank: chat rank (positive integer)
    :param alpha: alpha parameter, higher values bias towards sentiment (float in [0,1])
    :return: the chat sentimental value (float in [0,1])
    """

    # set global vars
    global sid
    global norm_dist
    global dist_normal_value

    # translate messages to english, might err on connections issues
    if translate_input:
        try:

            # init messages buffer
            translated_messages = []

            # translate messages
            for msg in messages:
                translated_messages.append(TextBlob(msg).translate())

            # set buffer to origin
            messages = translated_messages

        # in case of en->en translation, ignore err
        except NotTranslated:
            pass

        # in case of failed translation, prompt and exit (might err on connection issues)
        except TranslatorError as err:
            print "failed to translate messages:", err
            exit(1)

    # evaluate messages intensity values
    messages_neg_values = [sid.polarity_scores(message)['neg'] for message in messages]
    messages_pos_values = [sid.polarity_scores(message)['pos'] for message in messages]

    # calc the maximum of sentiment means
    chat_max_sentiment = max(mean(messages_neg_values), mean(messages_pos_values))

    # evaluate chat rank importance
    chat_rank = norm_dist.pdf(rank) / dist_normal_value

    # evaluate final chat rank as noted
    chat_value = (chat_max_sentiment * alpha) + (chat_rank * (1 - alpha))

    return chat_value


###############################################################################
# ------------------------- Mock Data & Evaluations ------------------------- #
###############################################################################

# import required evaluation modules
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# message VADER mock data
_messages = ["VADER is smart, handsome, and funny.",  # positive sentence example
             "VADER is smart, handsome, and funny!",
             # punctuation emphasis handled correctly (sentiment intensity adjusted)
             "VADER is very smart, handsome, and funny.",
             # booster words handled correctly (sentiment intensity adjusted)
             "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
             "VADER is VERY SMART, handsome, and FUNNY!!!",
             # combination of signals - VADER appropriately adjusts intensity
             "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",
             # booster words & punctuation make this close to ceiling for score
             "The book was good.",  # positive sentence
             "The book was kind of good.",  # qualified positive sentence is handled correctly (intensity adjusted)
             "The plot was good, but the characters are uncompelling and the dialog is not great.",
             # mixed negation sentence
             "A really bad, horrible book.",  # negative sentence with booster words
             "At least it isn't a horrible book.",  # negated negative sentence with contraction
             ":) and :D",  # emoticons handled
             "",  # an empty string is correctly handled
             "Today sux",  # negative slang handled
             "Today sux!",  # negative slang with punctuation emphasis handled
             "Today SUX!",  # negative slang with capitalization emphasis
             "Today kinda sux! But I'll get by, lol"
             # mixed sentiment example with slang and constrastive conjunction "but"
             ]


def evaluate_model():
    """
    evaluates the model's alpha-rank trade-off under a normal sentiment distribution
    pulled from the VADER model validation data-set and plots the said surface

    :return: None
    """

    # evaluate global vars
    global _messages

    # initialize figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # initialize mesh grid for rank and alpha
    X = np.arange(1, 100, 1)
    Y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(X, Y)

    # apply chat evaluation function to grid
    zs = np.array([evaluate_chat(_messages, x, y, True) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    # plot surface
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.YlOrRd, linewidth=0, antialiased=False)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Alpha (higher bias to sentiment)')
    ax.set_zlabel('Chat Value')
    ax.set_zlim(0, 1)

    # print plot
    plt.show()

evaluate_model()
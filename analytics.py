import pandas as pd
import numpy as np
import math

import time
import datetime



from wordcloud import WordCloud, ImageColorGenerator
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px

import seaborn as sns; sns.set(style='white')
from mpl_toolkits.mplot3d import Axes3D

def preprocess_to_squad(df):
    contexts = []
    answers = []
    toxic_words = []
    num_of_toxic_words = []

    for i in range(len(df)):
        text_str = df['spans'][i]
        splitted_str = text_str[1:-1].split(", ")
        if len(splitted_str) == 1:
            contexts.append(text_str)
            num_of_toxic_words.append(0)
        else:
            splitted_str = list(map(int, splitted_str))
            context = df['text'][i]
             gaps = [[s, e] for s, e in zip(splitted_str, splitted_str[1:]) if s + 1 < e]
            edges = iter(splitted_str[:1] + sum(gaps, []) + splitted_str[-1:])
            unformatted_answers = list(zip(edges, edges))

            for tokens in unformatted_answers:
                start_token, end_token = tokens[0], tokens[1]
                
                answer = {}
                answer["start_ans"] = start_token
                answer["end_ans"] = end_token + 1
                answer['ans'] = context[start_token:end_token + 1]
                toxic_words.append(context[start_token:end_token + 1])

                answers.append(answer)
                contexts.append(context)
            
            
            num_of_toxic_words.append(len(unformatted_answers))
    return contexts, answers, toxic_words, num_of_toxic_words

def plot_hist(df, postfix):
    ax = df.text.apply(lambda x: len(x.split())).hist(figsize = (12, 6));
    ax.set_ylabel("num of words");
    ax.set_xlabel("num of words in {postfix} texts");
    ax.set_title("Distribution of num of words in {postfix} texts")
    fig = ax.get_figure()
    fig.savefig("Distribution on {postfix}.png")
    return ax

def plot_toxic_hist(num_toxic):
    labels, counts = np.unique(num_toxic, return_counts=True)
    plt.figure(figsize=(15,6))
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    for label, count in zip(labels, counts):
        plt.text(x=label - 0.2 , y = count + 100 , s = count, fontdict=dict(fontsize=10))
    plt.tight_layout()

    plt.savefig("hist on {num_toxic} toxics.png")

def plot_word_cloud(corpus, name, toxic = 'toxic'):           
    plt.figure(figsize=(12,6))
    word_cloud = WordCloud(background_color='white', max_font_size = 80).generate(" ".join(corpus))
    plt.imshow(word_cloud)
    plt.title("WordCloud of {toxic} words from corpus {name}")
    plt.savefig("WordCloud of {toxic} words from corpus {name}.png")
    plt.axis('of')
    # # plt.show()
    
def corpus_from_text(text):
    corpus = []
    for sentence in text:
        for word in sentence:
            corpus.append(word)      
    return corpus

def corpus_from_toxic(toxic_text):
    corpus = []
    for sentence in toxic_text:
        if isinstance(sentence.split(), list):
            for word in sentence.split():
                corpus.append(word)
    return corpus

if __name__ == "__main__":
    url_test = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/tsd_test.csv"
    url_train = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/tsd_train.csv"
    url_trial = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/tsd_trial.csv"

    train_df = pd.read_csv(url_train, error_bad_lines=False)
    test_df = pd.read_csv(url_test, error_bad_lines=False)
    trial_df = pd.read_csv(url_trial, error_bad_lines=False)

    train_df = pd.read_csv(url_train, error_bad_lines=False)
    test_df = pd.read_csv(url_test, error_bad_lines=False)
    trial_df = pd.read_csv(url_trial, error_bad_lines=False)

    train_contexts, train_answers, train_toxic_words, num_train_toxic = preprocess_to_squad(train_df)
    val_contexts, val_answers, val_toxic_words, num_val_toxic = preprocess_to_squad(trial_df)
    test_contexts, test_answers, test_toxic_words, num_test_toxic = preprocess_to_squad(test_df)

    train_df_corpus = corpus_from_text(train_df.text.apply(lambda x: str(x).split()))
    val_df_corpus = corpus_from_text(trial_df.text.apply(lambda x: str(x).split()))
    test_df_corpus = corpus_from_text(test_df.text.apply(lambda x: str(x).split()))

    # data exploration for training set
    plot_hist(train_df, "training")
    print('The maximum number of words in the training set: ',  max(train_df.text.apply(lambda x: len(x.split()))))
    plot_word_cloud(train_df_corpus, "Dataset Exploration")
    plot_word_cloud(train_toxic_words, "Dataset Exploration")
    labels, counts = np.unique(num_train_toxic, return_counts=True)
    plt.figure(figsize=(15,6))
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    for label, count in zip(labels, counts):
        plt.text(x=label - 0.25, y = count + 50 , s = count, fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.xlabel('Number of toxic spans')
    plt.ylabel('Number of texts')
    plt.title('Distribution of toxic spans in the training texts')
    plt.savefig('Distribution of toxic spans in the training set.png')
    # plt.show()


    # data exploration for testing set
    print('The number of texts in test: ', len(test_df))
    plot_hist(test_df, 'testing')
    print('Maximum number of words in test sample: ',  max(test_df.text.apply(lambda x: len(x.split()))))
    labels, counts = np.unique(num_test_toxic, return_counts=True)
    plt.figure(figsize=(15,6))
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    for label, count in zip(labels, counts):
        plt.text(x=label - 0.1, y = count + 30 , s = count, fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.xlabel('Number of toxic spans')
    plt.ylabel('Number of texts')
    plt.title('Distribution of toxic spans in the test texts')
    plt.savefig('Distribution of toxic spans in the test set.png')
    # plt.show()
    plot_word_cloud(test_df_corpus, 'test')
    plot_word_cloud(test_toxic_words, 'test', toxic)

    # data exploration for validation set
    print('The number of texts in validation: ', len(trial_df))
    print('The maximum number of words in validation: ',  max(trial_df.text.apply(lambda x: len(x.split()))))
    plot_hist(trial_df, "validation");
    labels, counts = np.unique(num_val_toxic, return_counts=True)
    plt.figure(figsize=(15,6))
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    for label, count in zip(labels, counts):
        plt.text(x=label - 0.1, y = count + 5, s = count, fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.xlabel('Number of toxic spans')
    plt.ylabel('Number of texts')
    plt.title('Distribution of toxic spans in the validation texts')
    plt.savefig('Distribution of toxic spans in the validation set.png')
    # plt.show()
    plot_word_cloud(val_df_corpus, title)
    plot_word_cloud(val_toxic_words, title, toxic)



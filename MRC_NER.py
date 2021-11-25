import pandas as pd
import numpy as np
import mathimport time
import datetimeimport os
from transformers import AutoTokenizer, TFAutoModel, AutoConfig, TFBertModel
import tensorflow as tf
import pickle
from sklearn.model_selection import KFold
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import time
from datetime import datetime
import string
from counting_metrics import jaccard_score, f1_score

#avoid tensorflow print on standard error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
MAX_LEN = 256
model_name = 'albert-xxlarge-v2'
num_tags = 2

EPOCHS = 10 # originally 3
SEED = 88888

LABEL_SMOOTHING = 0.1
tf.random.set_seed(SEED)
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE

Dropout_new = 0.3 # originally 0.1
n_split = 2         # originally 5
lr = 5e-5           # originally 3e-5
num_words = 20

exp = True

strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE
# print("REPLICAS: ", strategy.num_replicas_in_sync)

def get_question(number_words):
    
    pattern = np.random.choice(patterns, 1)[0]
    flag = 'Does' in pattern
    
    toxic_words = set()

    while len(toxic_words) < number_words:
        if np.random.random() > 0.5:
            toxic_words.add(np.random.choice(synonyms, 1)[0])
        else:
            toxic_words.add(np.random.choice(dicriminations, 1)[0])
            
    question = pattern + ' ' + ', '.join(toxic_words) + '?'* flag
    
    return question

class NERTokenizerWords:
    
    def __init__(self, df, question, model_name, MAX_LEN, num_words = 25):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, use_fast = False)
        self.model_name = model_name
        self.MAX_LEN = MAX_LEN
        
        
        self.contexts, self.answers = self.get_contexts_answers(df)
        self.shape = len(self.contexts)
        
        if question == None:
            np.random.seed(SEED)
            self.questions = [get_question(num_words)  for _ in range(self.shape)]
        else:
            self.questions = [question] * self.shape
        
    def get_contexts_answers(self, df):
        contexts = []
        answers = []

        num_imp = 0

        for i in range(len(df)):

            text_str = df['spans'][i]
            splitted_str = text_str[1:-1].split(", ")
            context = df['text'][i]
            splitted_context = context.split()

            if len(splitted_str) == 1:
                contexts.append(context)
                answers.append([0]*len(splitted_context))
                num_imp += 1
                continue


            splitted_str = list(map(int, splitted_str))


            tags = [0]*len(splitted_context)
            char_tags = [0]*len(context)


            offsets = []; idx = 0;
            for word in splitted_context:
                idx = context.find(word, idx)
                offsets.append((idx, idx + len(word) + 1))
                idx += len(word)


            for pos in splitted_str:
                char_tags[pos] = 1

            for word_num, offset in enumerate(offsets):
                start_word, end_word = offset
                if sum(char_tags[start_word:end_word]) > 0:
                    tags[word_num] = 1


            contexts.append(' '.join(splitted_context)) 
            answers.append(tags)
            
        print("Number of impossible: ", num_imp)

        return contexts, answers

    def create_inputs_targets(self):
        
        roberta_flag = self.model_name == 'roberta-base' or self.model_name == 'roberta-large'
        
        dataset_dict = {
                "input_ids": [],
                "token_type_ids": [],
                "attention_mask": [],
                "tags": []
            }

        input_ids = []
        target_tags = []
        num_dropped = 0

        for context, question, answer in zip(self.contexts, self.questions, self.answers):
            input_ids = []
            target_tags = []
            for idx, word in enumerate(context.split()):
                if roberta_flag:
                    word = ' ' + word
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                input_ids.extend(ids)

                tokenized_words = self.tokenizer.tokenize(word)
                for tokenized_word in tokenized_words:
                    if tokenized_word in ',.!?':
                        target_tags.extend([0])
                    else:
                        target_tags.extend([answer[idx]])

            enc_question = self.tokenizer.encode(question, add_special_tokens=False)
            question_tags = [0]*len(enc_question)
            
            if roberta_flag:
                sep_tokens = 2
            else:
                sep_tokens = 1
                
            token_type_ids = [0] + question_tags + [0]*sep_tokens + [1] * len(input_ids) + [0]
            input_ids = [self.tokenizer.cls_token_id] + enc_question + [self.tokenizer.sep_token_id]*sep_tokens + input_ids + [self.tokenizer.sep_token_id]
            target_tags = [0] + question_tags + [0]*sep_tokens + target_tags + [0]
            attention_mask = [1] * len(input_ids)
            padding_len = self.MAX_LEN - len(input_ids)

            if padding_len < 0:
                num_dropped += 1
                continue

            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_len)
            attention_mask = attention_mask + ([self.tokenizer.pad_token_id] * padding_len)
            token_type_ids = token_type_ids + ([self.tokenizer.pad_token_id] * padding_len)
            target_tags = target_tags + ([2] * padding_len)


            dataset_dict["input_ids"].append(input_ids)
            dataset_dict["token_type_ids"].append(token_type_ids)
            dataset_dict["attention_mask"].append(attention_mask)
            dataset_dict["tags"].append(target_tags)


        for key in dataset_dict:
            dataset_dict[key] = np.array(dataset_dict[key])

        x = [
            dataset_dict["input_ids"],
            dataset_dict["token_type_ids"],
            dataset_dict["attention_mask"],
        ]
        y = dataset_dict["tags"]

        return x, y 
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

def masked_ce_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 2))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def create_model():
    ## BERT encoder
    config = AutoConfig.from_pretrained(model_name)
    encoder = TFAutoModel.from_pretrained(model_name, config = config)
    #alencoder = TFBertModel.from_pretrained(model_name, config = config, from_pt = True)

    ## NER Model
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    
    #embedding = tf.keras.layers.Dropout()(embedding)
    #tag_logits = tf.keras.layers.Dense(num_tags+1, activation='softmax')(embedding)
    
    embedding = tf.keras.layers.Dropout(Dropout_new)(embedding)
    embedding = tf.keras.layers.Conv1D(768, 2, padding='same')(embedding)
    embedding = tf.keras.layers.LeakyReLU()(embedding)
    embedding = tf.keras.layers.Conv1D(64, 2,padding='same')(embedding)
#     embedding = tf.keras.layers.Dense(1)(embedding)
#     embedding = tf.keras.layers.Flatten()(embedding)
    tag_logits = tf.keras.layers.Dense(num_tags + 1, activation = 'softmax')(embedding)
    
    model = tf.keras.models.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model

def predict(x, y_pred, y_real, prefix = 'TRAIN'):
    f1_scores, jaccard_scores = [], []
    predictions, true_answers = [], []
    shape = y_real.shape[0]
    input_ids = x[0]
    for i in range(shape):
        sep_token = np.where(input_ids[i] == tokenizer.sep_token_id)[0][-1]
        pred_tokens = np.where(y_pred[i] == 1)[0]
        pred_tokens = pred_tokens[pred_tokens < sep_token]
        pred_tokens = input_ids[i][pred_tokens]
        
        real_tokens = input_ids[i][np.where(y_real[i] == 1)[0]]
        
        pred_words = tokenizer.decode(pred_tokens)
        real_words = tokenizer.decode(real_tokens)
        
        predictions.append(pred_words)
        true_answers.append(real_words)
        
        jaccard_sc = jaccard_score(pred_words, real_words)
        f1_sc = f1_score(pred_words, real_words)
        
        f1_scores.append(f1_sc)
        jaccard_scores.append(jaccard_sc)
    
    print(f'{prefix} MEAN F1-SCORE {np.round(np.mean(f1_scores), 3)}')
    print(f'{prefix} MEAN JAC-SCORE {np.round(np.mean(jaccard_scores), 3)}')
    
    pred_csv = pd.DataFrame({'predictions': predictions, 'true_answers': true_answers, 'f1_scores': f1_scores, 'jaccard_scores': jaccard_scores})
    pred_csv.to_csv(f"{prefix}_{model_name.replace('/', '_')}_predictions.csv", index = False)
        
    return predictions, true_answers, f1_scores, jaccard_scores


if __name__ == "__main__":
    url_test = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/tsd_test.csv"
    url_train = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/tsd_train.csv"
    url_trial = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/tsd_trial.csv"

    train_df = pd.read_csv(url_train, error_bad_lines=False)
    test_df = pd.read_csv(url_test, error_bad_lines=False)
    trial_df = pd.read_csv(url_trial, error_bad_lines=False)

    synonyms = ['calumniation', 'insult', 'swearing', 'threat', 'discrimination', 
                'toxic words', 'severe toxic words', 'poisonous words', 
                'severe poisonous words', 'hate speech', 'offensive language', 
                'hatred', 'anger', 'violence', 'abuse', 'rudeness', 
                'profanity', 'cursing', 'intimidation', 'bullying', 
                'oppression', 'menace', 'stereotype', 'sexual harassment', 'hateful words'] 


    dicriminations = ['adultism', 'ageism', 'age discrimination',
                      'caste,ableism', 'disablism', 'disability discrimination', 
                      'linguistic discrimination', 'racism', 'racial discrimination', 
                      'discrimination based on skin colour', 'ethnic discrimination', 
                      'racial segregation', 'religious bigotry', 'religious discrimination',
                      'sexism', 'homophobia', 'misogyny', 'misandry', 'transphobia', 
                      'biphobia', 'lookism', 'antisemitism', 'hispanophobia', 
                      'islamophobia', 'sizeism', 'xenophobia', 'chauvinism', 
                      'afrophobia', 'anti-arabism', 'apostasy', 'colourism', 
                      'heightism', 'discrimination against intersex people', 
                      'supremacism', 'genetic discrimination', 'mentalism', 
                      'antisexualism', 'anti-Catholicism'] 

    patterns = ['Does the text include toxicity, such as', 
                'Does the text include',
                'Select spans of toxicity in the text, such as', 
                'Find spans of toxicity in the text, such as',
                'Find in the text spans of toxicity, such as']

    model = create_model()
    copy_train_df = train_df
    question = "Does the text include calumniation, insult, swearing, threat, discrimination, toxic or severe toxic words?"

    if exp:
        train_questions, val_questions, test_questions = None, None, None
    else:
        train_questions, val_questions, test_questions = question, question, question
        
    train_tokenized = NERTokenizerWords(copy_train_df, train_questions, model_name, MAX_LEN, num_words)
    tokenizer = train_tokenized.tokenizer
    x_train, y_train = train_tokenized.create_inputs_targets()


    val_tokenized = NERTokenizerWords(trial_df, val_questions, model_name, MAX_LEN, num_words)
    x_val, y_val = val_tokenized.create_inputs_targets()

    test_tokenized = NERTokenizerWords(test_df, test_questions, model_name, MAX_LEN, num_words)
    x_test, y_test = test_tokenized.create_inputs_targets()

    BATCH_SIZE = 256 if tpu else 16

    model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        verbose = 1,
        batch_size = BATCH_SIZE,
        validation_split = 0.1
    )
        

    train_predictions = model.predict(x_train, verbose = True)
    train_pred_tags = np.argmax(train_predictions, 2)

    val_predictions = model.predict(x_val, verbose = True)
    val_pred_tags = np.argmax(val_predictions, 2)

    train_predictions, train_true_values, tr_f1_scores, tr_jaccard_scores = predict(x_train, train_pred_tags, y_train, prefix='train')
    val_predictions, val_true_values, val_f1_scores, val_jaccard_scores = predict(x_val, val_pred_tags, y_val, prefix='val')

    test_predictions = model.predict(x_test, verbose = True)
    test_pred_tags = np.argmax(test_predictions, 2)

    test_predictions, test_true_values, test_f1_scores, test_jaccard_scores = predict(x_test, test_pred_tags, y_test, prefix='test')

    model.fit(
        x_val,
        y_val,
        epochs = EPOCHS // 3,
        verbose = 1,
        batch_size = BATCH_SIZE,
        validation_split = 0.1
    )


    test_predictions = model.predict(x_test, verbose = True)
    test_pred_tags = np.argmax(test_predictions, 2)

    test_predictions, test_true_values, test_f1_scores, test_jaccard_scores = predict(x_test, test_pred_tags, y_test, prefix='TEST')

    print(f'{model_name} finished')
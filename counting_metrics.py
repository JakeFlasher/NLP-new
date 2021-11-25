from os import walk
import os
import pandas as pd
import numpy as np

def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

def jaccard_score(pred, gold): 
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if (len(pred_tokens)==0) & (len(gold_tokens)==0): 
        return 0.5
    inter_tokens = pred_tokens.intersection(gold_tokens)
    return float(len(inter_tokens)) / (len(pred_tokens) + len(gold_tokens) - len(inter_tokens))


def f1_score(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens) 
    common_tokens = set(pred_tokens) & set(gold_tokens)
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(gold_tokens)
    return 2 * (prec * rec) / (prec + rec)

def get_result(path):
    results = {}
    approach = path.split('/')[-3]
    method = path.split('/')[-2]
    for r, d, f in os.walk(path):
        for file in f:
            #print(os.path.join(r, file))
            model_name = r.split('/')[-1]
            df_name = file.split('_')[0]
            #print(model_name, df_name)
            temp_df = pd.read_csv(os.path.join(r, file))
            temp_df['predictions'] = temp_df['predictions'].astype(str)
            before = temp_df.f1_scores.mean()
            for token in tokenizer_symbols[model_name]:
                #print(model_name, token, len(temp_df[temp_df['predictions'] == token]))
                temp_df['new_predictions'] = temp_df.apply(lambda x: x['predictions'].replace(token, 'impossible')\
                             if len(x['predictions'].split()) == 1 else x['predictions'], axis = 1)

            temp_df['new_f1'] = temp_df.apply(lambda x: f1_score(str(x['new_predictions']), str(x['true_answers'])), axis = 1)
            temp = temp_df.groupby('train_contexts').agg({'new_predictions': lambda x: ' '.join(x), \
                                                          'true_answers': lambda x: ' '.join(x)}).reset_index()
            f1_scores = temp.apply(lambda x: f1_score(x['new_predictions'], x['true_answers']), axis = 1)
            temp_df.name = model_name + ' ' + df_name
            if bool(results.get(model_name)):
                results[model_name][df_name] = np.round(f1_scores.mean(), 3) 
                # [before, temp_df.f1_scores.mean(), f1_scores.mean()] 
            else:
                results[model_name] = {df_name: np.round(f1_scores.mean(), 3)}
                # [before, temp_df.f1_scores.mean(), f1_scores.mean()]} 
    df = pd.DataFrame(results).T
    df.name = approach + ' ' + method
    return df

def get_ner_result(path):
    results = {}
    approach = path.split('/')[-3]
    method = path.split('/')[-2]
    for r, d, f in os.walk(path):
        for file in f:
            #print(os.path.join(r, file))
            model_name = r.split('/')[-1]
            df_name = file.split('_')[0]
            temp_df = pd.read_csv(os.path.join(r, file))
            temp_df['predictions'] = temp_df['predictions'].astype(str)
            if bool(results.get(model_name)):
                results[model_name][df_name] = np.round(temp_df.f1_scores.mean(), 3) 
                # [before, temp_df.f1_scores.mean(), f1_scores.mean()]
            else:
                results[model_name] = {df_name: np.round(temp_df.f1_scores.mean(), 3)}
                # [before, temp_df.f1_scores.mean(), f1_scores.mean()]}
    df = pd.DataFrame(results).T
    df.name = approach + ' ' + method
    return df

def train_val_test_datasets(models):
    names = []
    train, val, test = [], [], []
    for df in models:
        if 'train' in df.columns:
            train.append(df['train'])
        else:
            train.append(df['TRAIN'])
        if 'valid' in df.columns:
            val.append(df['valid'])
        elif 'VAL' in df.columns:
            val.append(df['VAL'])
        else:
            val.append(df['val'])
        if 'test' in df.columns:
            test.append(df['test'])
        else:
            test.append(df['TEST']) 
        names.append(df.name)
    train = pd.concat(train, axis = 1)
    val = pd.concat(val, axis = 1)
    test = pd.concat(test, axis = 1)
    for df in [train, val, test]:
        df.columns = names
        df['Average results of different Models'] = df.mean(axis = 1)
        df.loc["Average results by approaches"] = df.mean(axis = 0)
    return train, val, test

if __name__ == "__main__":
    files = []
    # path = 'results/impossible/MRC/not_tuned/base' 
    # for (dirpath, dirnames, filenames) in walk(path):
    #     files.extend(filenames)
    # path = 'results/'    
    imp_mrc_no_squad_base = []    
    tokenizer_symbols = {'BERT': ['[SEP]', '[PAD]' ], 'ALBERT': ['[SEP]', '[PAD]'], 
                         'SpanBERT': ['[SEP]', '[PAD]'], 'XLM-RoBERTa': ['nan', '<pad>', '</s>'], 
                         'RoBERTa': ['nan', '<pad>', '</s>'], 
                         'XXlargeALBERT': ['[SEP]', '[PAD]'], 'XlargeALBERT': ['[SEP]', '[PAD]']}
    # Pulling data on the MRC approach 
    imp_mrc_no_base = 'results/impossible/MRC/not_tuned/base'
    imp_mrc_no_large = 'results/impossible/MRC/not_tuned/large'    
    imp_mrc_squad_base = 'results/impossible/MRC/squad/base'
    imp_mrc_squad_large = 'results/impossible/MRC/squad/large'    
    imp_mrc_squad_gq_base = 'results/impossible/MRC/squad_QG/base'
    imp_mrc_squad_gq_large = 'results/impossible/MRC/squad_QG/large'    
    imp_mrc_no_base_df = get_result(imp_mrc_no_base)
    imp_mrc_no_large_df = get_result(imp_mrc_no_large)    
    imp_mrc_squad_base_df = get_result(imp_mrc_squad_base)
    imp_mrc_squad_large_df = get_result(imp_mrc_squad_large)
    imp_mrc_squad_gq_base_df = get_result(imp_mrc_squad_gq_base)
    imp_mrc_squad_gq_large_df = get_result(imp_mrc_squad_gq_large)

    # Pulling data on MRC + NER approach
    imp_mrc_ner_base = 'results/impossible/MRC+NER/no_QG/base'
    imp_mrc_ner_large = 'results/impossible/MRC+NER/no_QG/large'              
    imp_mrc_ner_qg_base = 'results/impossible/MRC+NER/QG/base'
    imp_mrc_ner_qg_large = 'results/impossible/MRC+NER/QG/large'
    imp_mrc_ner_base_df = get_ner_result(imp_mrc_ner_base)
    imp_mrc_ner_large_df = get_ner_result(imp_mrc_ner_large)    
    imp_mrc_ner_qg_base_df = get_ner_result(imp_mrc_ner_qg_base)
    imp_mrc_ner_qg_large_df = get_ner_result(imp_mrc_ner_qg_large)

    # pull out data on not impossible questions:
    no_imp_large = 'results/plausible/large'
    no_imp_base = 'results/plausible/base'    
    no_imp_large_df = get_result(no_imp_large)
    no_imp_base_df = get_result(no_imp_base)

    # look at all the results
    base_models = [imp_mrc_no_base_df, imp_mrc_squad_base_df,  imp_mrc_squad_gq_base_df, imp_mrc_ner_base_df, imp_mrc_ner_qg_base_df]
    large_models = [imp_mrc_no_large_df, imp_mrc_squad_large_df, imp_mrc_squad_gq_large_df, imp_mrc_ner_large_df, imp_mrc_ner_qg_large_df]
    pd.concat(base_models, axis = 1)
    pd.concat(large_models, axis = 1)    

    train_base, val_base, test_base = train_val_test_datasets(base_models)
    mean = train_base.mean(axis = 0)
    mean.name = 'Average by Approach'

    train_base = train_base.append(mean)
    train_base.style.apply(highlight_max)
    val_base.style.apply(highlight_max)
    test_base.style.apply(highlight_max)
    train_large, val_large, test_large = train_val_test_datasets(large_models)
    train_large.style.apply(highlight_max)
    val_large.style.apply(highlight_max)
    test_large.style.apply(highlight_max)


    train_base.to_csv('base_train.csv')
    val_base.to_csv('base_val.csv')
    test_base.to_csv('base_test.csv')
    train_large.to_csv('large_train.csv')
    val_large.to_csv('large_val.csv')
    test_large.to_csv('large_test.csv')
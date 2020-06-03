import os, glob
import pandas as pd
import random

def create_df(file, sep):
    """Return a DataFrame from a file.

    Args:
        file (str): a target file.
        sep (str): a separator for 'csv' or 'tsv'.
    Return:
        df (pd.DataFrame): df with column-names.
    """
    if os.path.isfile(file):
        try:
            #df = pd.read_csv(file, sep, encoding="shift_jis")
            df = pd.read_csv(file, sep)
            print("reading done for {}".format(file))
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep, encoding="cp932") # for case8!.csv
            print("reading done for {}".format(file))
    else:
        print("{} is nothing on your directory.".format(file))
        exit()
    return df

def add_context(df, num_pre_context, nan_key="<NA>"):
    """Add a context for the target dialog.

    Args:
        df (pd.DataFrame): df by create_df().
        num_pre_context (int): the number of dialogues before the target one.
        nan_key (str): a specified string to use when there is no previous dialog.
    Returns:
        result (pd.DataFrame): a new dataframe with a context in 'dialogue'
    """
    if num_pre_context <= 0:
        return df
    
    result = df.copy()
    for i in range(len(df)):
        pre_context = ""
        pre_index = list(range(1,num_pre_context+1))
        pre_index.reverse()
        for relative_index in pre_index:
            target_index = i - relative_index
            if target_index < 0:
                pre_context += nan_key + "\t"
                continue
            elif target_index >= len(df):
                print("something wrong: target_index={}, len(df)={}".format(target_index,len(df)))
                exit()
            else:
                pre_context += df.at[target_index, 'dialogue'] + "\t"
        new_body = pre_context + df.at[i, 'dialogue']
        result.at[i, 'dialogue'] = new_body
    return result

def load_MITI_dialog(directory, extension="csv", num_pre_context=0):
    """Load a dataset (mutli-files) for MITI dialog.

    Args:
        directory (str): a relative path.
        extension (str): 'csv' or 'tsv'.
        num_pre_context (int): the number of dialogues before the target one.
        #num_post_context (int): the number of dialogues after the target one.
    Returns:
        all_df (dict): a set of dataframe made by add_context(). the key is filename.
        documents_df (dict): a set of dataframe including a pair of dialogue and filename only.
    
    >>> all_df, documents_df = load_MITI_dialog('../dataset/example-20200312/', 'csv', 1)
    reading done for ../dataset/example-20200312/case5.csv
    reading done for ../dataset/example-20200312/case4.csv
    reading done for ../dataset/example-20200312/case1.csv
    reading done for ../dataset/example-20200312/case3.csv
    reading done for ../dataset/example-20200312/case2.csv
    >>> len(all_df)
    5
    >>> all_df['../dataset/example-20200312/case1.csv'].at[0, 'dialogue']
    '<NA>\\t今回はどのような目的で来られましたか？'
    >>> all_df['../dataset/example-20200312/case1.csv'].at[0, 'MITI_code']
    0
    """
    if extension == "csv" or extension == "tsv":
        target = directory + "/*." + extension
        if extension == "csv":
            sep = ','
        else:
            sep = '\t'
    else:
        print("Extension must be 'csv' or 'tsv'")
        exit()
    
    files = glob.glob(target)
    if len(files) == 0:
        print("There is no dataset on {}/".format(directory))
        exit()
    
    all_df = {file: create_df(file, sep) for file in files}
    
    # ready for normal dialogues for preprocessing
    documents_df = {}
    for key in all_df.keys():
        dialogues = ''
        for utterance in all_df[key]['dialogue']:
            dialogues += utterance + '\n'
        documents_df[key] = dialogues

    # read for dialogues with context
    for key in all_df.keys():
        df = add_context(all_df[key], num_pre_context)
        all_df[key] = df
    return all_df, documents_df

def dfs_to_dataset(all_df, keys):
    """Extract a specified dialogue with the keys.

    Args:
        all_df (dict): a set of df created by load_MITI_dialog().
        keys (list): a set of key to extract from all_df for training dataset.
    Returns:
        dialogues (list): a set of a dialogue contains of context.
        signals (list): a set of teacher signal (MITI_code).
    """
    dialogues = []
    signals = []
    for key in keys:
        #for df in all_df[key]:
            for index in range(len(all_df[key])):
                dialogues.append(all_df[key].at[index, 'dialogue'])
                signals.append(all_df[key].at[index, 'MITI_code'])
    return dialogues, signals

def divide_dialog(all_df, train_rate=0.8, seed=None):
    """Divide a dataset into train & test set with train_rate.
    To avoid cheating, this function process on a discource basis
    rather than on a utterance basis. This selection is made at random.

    Note:
        The train_rate is NOT directly related to the number of samples.
        It is used to determine the value of discources. For example,
        lets assume there are 5 discources that each discource consists of
        100, 200, 300, 400 and 500 utterances (samples). When we determine
        to use the first 3 discources for trainting set, the number of
        training samples is 100 + 200 + 300 = 600 and the number of test
        samples is 400 + 500 = 900, even though the train_rate = 0.6.

    Args:
        all_df (dict): a set of df created by load_MITI_dialog().
        train_rate (float): a value (max=1.0) to set training dataset.
        seed (int): you should set a seed when you want to fix the shuffled results.
    Returns:
        X_train, X_test, y_train, y_test (list):
            X is a list of dialogue.
            y is a list of MITI_code.
    
    >>> all_df, documents_df = load_MITI_dialog('../dataset/example-20200312/', 'csv', 1)
    reading done for ../dataset/example-20200312/case5.csv
    reading done for ../dataset/example-20200312/case4.csv
    reading done for ../dataset/example-20200312/case1.csv
    reading done for ../dataset/example-20200312/case3.csv
    reading done for ../dataset/example-20200312/case2.csv
    >>> X_train, X_test, y_train, y_test = divide_dialog(all_df, train_rate=0.8, seed=0)
    >>> print(len(X_train), len(y_train))
    20 20
    >>> print(len(X_test), len(y_test))
    5 5
    >>> X_train[0]
    '<NA>\\t今回はどのような目的で来られましたか？'
    >>> y_train[0]
    0
    """
    if seed != None:
        random.seed(seed)

    # divide dataset into train & test
    num_train_samples = int(len(all_df) * train_rate)
    all_sample_keys = list(all_df.keys())
    random.shuffle(all_sample_keys)
    train_keys = all_sample_keys[:num_train_samples]
    test_keys = all_sample_keys[num_train_samples:]

    # ready for dialogue and signal
    X_train, y_train = dfs_to_dataset(all_df, train_keys)
    X_test, y_test = dfs_to_dataset(all_df, test_keys)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    all_df, documents_df = load_MITI_dialog("../dataset/example-20200312/", "csv", num_pre_context=1)
    #print(all_df['../dataset/example-20200312/case1.csv'].at[0, 'dialogue'])
    #print(all_df['../dataset/example-20200312/case1.csv'].at[0, 'MITI_code'])
    print(documents_df['../dataset/example-20200312/case2.csv'])

    X_train, X_test, y_train, y_test = divide_dialog(all_df, train_rate=0.8)
    #print(len(X_train), len(y_train))
    #print(len(X_test), len(y_test))
    #print(X_train[0])
    #print(y_train[0])

import os, glob
import pandas as pd

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
            df = pd.read_csv(file, sep, encoding="shift_jis")
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
    for key in all_df.keys():
        df = add_context(all_df[key], num_pre_context)
        all_df[key] = df
    return all_df

if __name__ == '__main__':
    all_df = load_MITI_dialog("../dataset/ver-20200312/", "csv", num_pre_context=1)
    print(all_df['../dataset/ver-20200312/case7.csv'].at[41, 'dialogue'])
    print(all_df['../dataset/ver-20200312/case7.csv'].at[41, 'MITI_code'])

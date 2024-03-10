'''
exploreData
To explore and find patterns within the text data.
'''
import os
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# jsonPath - where train.json file exists.
rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jsonPath = os.path.join(rootDir, 'data', 'train.json')

def exploreData(jsonPath = jsonPath):
    '''
    exPloreData
    Args:
        jsonPath = location of the JSON dataset.
    Returns:
        None
    '''
    # Remove pandas filter warnings
    warnings.filterwarnings('ignore', 
                            category = pd.errors.SettingWithCopyWarning)
    
    # If JSON file exists, load the data
    if os.path.exists(jsonPath):
        trainJSON = json.load(open(jsonPath))
        train = pd.json_normalize(trainJSON)
        print('SUCCESS! train.json loaded')
    else:
        print(f'FAILED! No data at the specified location {jsonPath}')
        raise FileNotFoundError(f'FAILED! No data at the specified location {jsonPath}')

    # Tokenizing
    tokens = defaultdict(list)
    for ts, ls in zip(train["tokens"], train["labels"]):
        for t, l in zip(ts, ls):
            if l != "O":
                tokens[l.split("-")[1]].append(t)
    
    # Unique labels in tokens.
    print('Unique labels: ',
          {k: len(set(v)) for k,v in tokens.items()})

    # Create path to logging plots if not present already
    plotsPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'plots'))
    if not os.path.exists(plotsPath):
        # Create the folder if it doesn't exist
        os.makedirs(plotsPath)
        print(f"Folder '{plotsPath}' created successfully.")
    
    # tokenCount to fetch token lengths
    train["tokens"].apply(len).plot.hist(bins = 50) # Generate hist plots.
    plt.xlabel("Number of Tokens in Documents")
    plt.title("Token Length for All Documents")
    plt.savefig(os.path.join(plotsPath, 'Token_Lengths_of_Documents.png'))
    
    full_ner_labels = [
    'B-NAME_STUDENT', 'I-NAME_STUDENT',
    'B-URL_PERSONAL', 'I-URL_PERSONAL',
    'B-ID_NUM', 'I-ID_NUM',
    'B-EMAIL', 'I-EMAIL',
    'B-STREET_ADDRESS', 'I-STREET_ADDRESS',
    'B-PHONE_NUM', 'I-PHONE_NUM',
    'B-USERNAME', 'I-USERNAME']

    # Data stats.
    train_eda = train.copy()
    # Find documnets with high number of entities
    train_eda['ner_labels'] = train_eda['labels'].apply(lambda x: [item for item in x if item != 'O'])
    train_eda['count_ner_labels'] = train_eda['ner_labels'].apply(len)
    train_eda['count_distinct_ner_labels'] = train_eda['ner_labels'].apply(lambda x: len(set(x)))
    train_eda.sort_values(by='count_distinct_ner_labels', inplace=True, ascending=False)

    exploded_df = train_eda['ner_labels'].explode()
    dummies = pd.get_dummies(exploded_df).reset_index()
    frequency = dummies.sum().sort_values(ascending=False)
    ordered_columns = frequency.index.tolist() ; ordered_columns.remove("index")
    counted = dummies.groupby('index').sum()
    counted = counted.reindex(columns=full_ner_labels, fill_value=0)
    counted = counted[ordered_columns + [i for i in full_ner_labels if i not in ordered_columns]]

    train_eda = train_eda.join(counted)

    # NER: NER labels data distribution
    num_documents = train_eda.shape[0]
    ner_labels_data = train_eda[full_ner_labels].melt(var_name='ner_label', value_name='count')
    ner_labels_stat = ner_labels_data.groupby('ner_label').agg(
        doc_count=pd.NamedAgg(column='count', aggfunc=lambda x: (x > 0).sum()),
        ner_count=pd.NamedAgg(column='count', aggfunc="sum"),).reset_index()
    ner_labels_stat['doc_count_percentage'] = np.round(ner_labels_stat['doc_count'] /num_documents,4)
    ner_labels_stat['ner_count_percentage'] = np.round(ner_labels_stat['ner_count'] /sum(ner_labels_stat['ner_count']),4)
    ner_labels_stat = ner_labels_stat.sort_values('doc_count', ascending = False)

    ner_labels_stat.to_csv(os.path.join(plotsPath, 'ner_labels_stats.csv'), index = False)

    def plot_ner_distribution(ner_labels_stat, count_col, percentage_col):
        import matplotlib.colors as mcolors
        plt.figure(figsize=(16,8))
        unique_labels = ner_labels_stat['ner_label'].unique()
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels)))
        color_dict = dict(zip(unique_labels, colors))
        
        for label in unique_labels:
            subset = ner_labels_stat[ner_labels_stat['ner_label'] == label]
            plt.bar(subset['ner_label'], subset[count_col], color=color_dict[label])
            
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Create secondary y-axis for percentage
        sec_axis = plt.twinx()
        sec_axis.plot(ner_labels_stat['ner_label'], ner_labels_stat[percentage_col], color='r')
        sec_axis.set_ylabel('Percentage')
        
        # Titles and labels
        plt.title('Count / Percentage of NER Labels')
        plt.xlabel('NER Label')
        plt.savefig(os.path.join(plotsPath, 'ct_pt_of_ner_labels.png'))
    
    plot_ner_distribution(ner_labels_stat, 'doc_count', 'doc_count_percentage')

# exploreData(jsonPath = jsonPath)
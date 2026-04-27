import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

# CONFIGURATION
PROJECT = 'pytorch'  # options: pytorch, tensorflow, keras, incubator-mxnet, caffe
REPEAT = 30
out_csv_name = f'{PROJECT}_LR.csv'

# TEXT PREPROCESSING
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def preprocess(text):
    return clean_str(remove_emoji(remove_html(text)))

# DATA LOADING
def load_dataset(project):
    pd_all = pd.read_csv(f'{project}.csv')
    pd_all = pd_all.sample(frac=1, random_state=999)

    pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
    )

    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])
    
    data = pd.read_csv('Title+Body.csv').fillna('')
    data['text'] = data['text'].apply(preprocess)
    return data 

# EMBEDDING
def generate_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=True)

def experiment(embeddings, labels, REPEAT):
    
    accuracies  = []
    precisions  = []
    recalls     = []
    f1_scores   = []
    auc_values  = []

    for repeated_time in range(REPEAT):
        # split into train/test
        indices = np.arange(len(labels))
        train_index, test_index = train_test_split(
            indices, test_size=0.3, random_state=repeated_time
        )

        train_text = embeddings[train_index]
        test_text = embeddings[test_index]

        y_train = labels.iloc[train_index]
        y_test = labels.iloc[test_index]

        # logistic regression and gridsearch
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000),
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=5,
            scoring='f1_macro' # using f1 macro as the metric for seleection - f1 is balance of precision and recall, making it good for imbalanced datasets
        )
        grid.fit(train_text, y_train)
        #retireve the best model
        best_clf = grid.best_estimator_
        best_clf.fit(train_text, y_train)

        # make predictions
        y_pred = best_clf.predict(test_text)
        y_prob = best_clf.predict_proba(test_text)[:, 1]

        #evaluation
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        auc_values.append(roc_auc_score(y_test, y_prob))

    #aggregate the results
    final_accuracy  = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall    = np.mean(recalls)
    final_f1        = np.mean(f1_scores)
    final_auc       = np.mean(auc_values)

    print("\nSentence Transformer + Logistic Regression")
    print(f"Number of repeats: {REPEAT}")
    print(f"Average Accuracy:  {final_accuracy:.4f}")
    print(f"Average Precision: {final_precision:.4f}")
    print(f"Average Recall:    {final_recall:.4f}")
    print(f"Average F1 score:  {final_f1:.4f}")
    print(f"Average AUC:       {final_auc:.4f}")

    df_log = pd.DataFrame({
        'repeated_times': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
        'AUC': [final_auc],
        'CV_list(Accuracy)': [str(accuracies)],
        'CV_list(Precision)': [str(precisions)],
        'CV_list(Recall)': [str(recalls)],
        'CV_list(F1)': [str(f1_scores)],
        'CV_list(AUC)': [str(auc_values)]
    })
    #saving results to csv
    try:
        pd.read_csv(out_csv_name, nrows=1)
        df_log.to_csv(out_csv_name, mode='a', header=False, index=False)
    except:
        df_log.to_csv(out_csv_name, mode='w', header=True, index=False)

    print(f"Results saved to: {out_csv_name}")

# MAIN
def main():
    data = load_dataset(PROJECT)
    embeddings = generate_embeddings(data['text'].tolist())
    experiment(embeddings, data['sentiment'], REPEAT)

if __name__ == '__main__':
    main()
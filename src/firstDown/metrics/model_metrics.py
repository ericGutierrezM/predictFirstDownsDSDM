from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='binary')

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='binary')

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary')

def roc_auc(y_true, y_pred_prob):
    return roc_auc_score(y_true, y_pred_prob)
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_model(model='lgbm'):
    match model:
        case 'lgbm':
            clf = LGBMClassifier()
        case 'logreg':
            clf = LogisticRegression(max_iter=1000)
        case 'knn':
            clf = KNeighborsClassifier()
        case 'dtree':
            clf = DecisionTreeClassifier()
        case 'rf':
            clf = RandomForestClassifier()
        case 'gb':
            clf = GradientBoostingClassifier()
        case 'hgb':
            clf = HistGradientBoostingClassifier()
        case 'xgb':
            clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        case 'cat':
            clf = CatBoostClassifier(verbose=0)
        case _:
            raise ValueError(f"Unknown model: {model}")

    return clf

def do_predict(X, clf):
    y_pred = clf.predict(X)
    y_pred_prob = clf.predict_proba(X)[:,1]
    return y_pred, y_pred_prob

def do_fit(random_search, X, y):
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    return best_model
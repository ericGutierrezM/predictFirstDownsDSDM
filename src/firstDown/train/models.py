from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def fit(X, y, model='lgbm'):
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

    clf.fit(X, y)
    return clf
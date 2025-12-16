from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def r_search(param_dist, model):

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,               # number of random combinations
        cv=3,                    # cross-validation folds
        scoring="accuracy",
        random_state=11,
        n_jobs=-1,               # use all cores
        verbose=2
    )
    return random_search

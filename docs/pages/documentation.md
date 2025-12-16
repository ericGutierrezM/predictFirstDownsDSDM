This page documents all the functionalities in the _firstDown_ library. We make use of the sublibraries to structure them.

### 1. feature_engineering
---
```.defense_pass(dataset)```

This function, specific to the implementation for the prediction of first downs in the NFL, creates a new feature that captures the quailty of the defense when facing an offensive pass play, based on their performance against this kind of plays in previous weeks.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.build_features.defense_pass(dataset)
---
```.defense_rush(dataset)```

This function, specific to the implementation for the prediction of first downs in the NFL, creates a new feature that captures the quailty of the defense when facing an offensive rush play, based on their performance against this kind of plays in previous weeks.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.build_features.defense_rush(dataset)
---
```.defense_scramble(dataset)```

This function, specific to the implementation for the prediction of first downs in the NFL, creates a new feature that captures the quailty of the defense when facing a QB scramble, based on their performance against this kind of plays in previous weeks.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.build_features.defense_scramble(dataset)
---
```.get_positions(dataset_play_by_play, dataset_players)```

This function, specific to the implementation for the prediction of first downs in the NFL, creates three new features that capture the position of the players that ran, passed, or received the ball in a given play.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.build_features.get_positions(dataset_play_by_play, dataset_players)
---
```.inertia(dataset)```

This function, specific to the implementation for the prediction of first downs in the NFL, creates a new feature that captures how well did the last 3 offensive series go, based on the variable ```series_success```. Note that the combination of these series is a weighted average.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.build_features.inertia(dataset)
---
```.one_hot(dataset, cols)```

This function creates a one hot encoder with the training data that will later be used to encode features on both the train and test datasets.

Returns a ```class```.

        import firstDown
        encoder = firstDown.feature_engineering.encode.one_hot(X_train, one_hot_cols)
---
```.one_hot_transform(dataset, cols, encoder)```

This function takes the encoder created with ```.one_hot()``` and transforms the specified features.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.encode.one_hot_transform(X_train, one_hot_cols, encoder)
---
```.play_type(dataset)```

This function, specific to the implementation for the prediction of first downs in the NFL, creates a new feature that captures the play type of the offense. It can be 'pass', 'rush', or 'scramble'.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.feature_engineering.build_features.play_type(dataset)
---
---
### 2. hyper_tuning
---
```.r_search(param_dist, model)```

This function implements hyperparameter tuning using a randomized search.

Returns a ```class```.

        import firstDown
        search = firstDown.hyper_tuning.random_search.r_search(param_dist, model)
---
---
### 3. load_data
---
```.nfl_data()```

This function imports the play-by-play and players datasets for the NFL from ```nflreadpy´´´.

Returns two ```DataFrame```.

        import firstDown
        pbp, players = firstDown.load_data.datasets.nfl_data()
---
---
### 4. metrics
---
```.accuracy(y_true, y_pred)```

This function returns the accuracy score of the model computed using the predicted and true values for the target.

Returns a ```float```.

        import firstDown
        accuracy = firstDown.metrics.model_metrics.accuracy(y_test, y_pred)
---
```.f1(y_true, y_pred)```

This function returns the F1 score of the model computed using the predicted and true values for the target.

Returns a ```float```.

        import firstDown
        f1 = firstDown.metrics.model_metrics.f1(y_test, y_pred)
---
```.precision(y_true, y_pred)```

This function returns the precision score of the model computed using the predicted and true values for the target.

Returns a ```float```.

        import firstDown
        precision = firstDown.metrics.model_metrics.precision(y_test, y_pred)
---
```.recall(y_true, y_pred)```

This function returns the recall score of the model computed using the predicted and true values for the target.

Returns a ```float```.

        import firstDown
        recall = firstDown.metrics.model_metrics.recall(y_test, y_pred)
---
```.roc_auc(y_true, y_pred_prob)```

This function returns the ROC AUC score of the model computed using the true values and the predicted probabilities for the target.

Returns a ```float```.

        import firstDown
        roc_auc = firstDown.metrics.model_metrics.roc_auc(y_test, y_pred_prob)
---
---
### 5. preprocessing
---
```.drop_control_rows(dataset, control_col, filter_out)```

This function, specific to the implementation for the prediction of first downs in the NFL, drops the rows that are only used for control purposes and have no information about the plays.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.preprocessing.clean.drop_control_rows(dataset, control_col='desc')
---
```.drop_nan(dataset, cols)```

This function drops the rows with one or more NaN values in the specified columns.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.preprocessing.deal_nan.drop_nan(dataset, cols)
---
```.drop_penalties(dataset, penalty_col)```

This function, specific to the implementation for the prediction of first downs in the NFL, drops the rows where the a first down resulted through a penalty.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.preprocessing.clean.drop_penalties(dataset, penalty_col='first_down_penalty')
---
```.replace_nan(dataset, cols, method, num, txt)```

This function replaces the NaN values in a subset of columns by a specified value, which varies depending on the method selected.

Returns both a ```DataFrame``` and a value, whose type will depend on the method selected.

        import firstDown
        updated_dataset, value = firstDown.preprocessing.deal_nan.replace_nan(dataset, cols, method, num, txt)
---
```.scaler(dataset, num_cols)```

This function returns the scaler for the numerical columns specified.

Returns a ```class```.

        import firstDown
        updated_dataset = firstDown.preprocessing.scale.scaler(X_train, num_cols)
---
```.scaler_transform(dataset, num_cols, scaler)```

This function scales the numerical columns specified.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.preprocessing.scale.scaler(X_train, num_cols, scaler)
---
```.search_nan(dataset)```

This function generates a table with the number of rows with NaN values for each feature.

Returns a ```DataFrame```.

        import firstDown
        updated_dataset = firstDown.preprocessing.deal_nan.search_nan(dataset)
---
```.split_data(dataset, y_col, test_size)```

This function splits a dataset into X and y, and into train and test.

Returns four ```DataFrame```s.

        import firstDown
        X_train, X_test, y_train, y_test = firstDown.preprocessing.split.split_data(dataset, y_col, test_size)
---
---
### 6. train
---
```.do_fit(search, X, y)```

This function fits and returns the best estimator from the hyperparameter tuning.

Returns a ```class```.

        import firstDown
        best_model = firstDown.train.models.do_fit(search, X_train, y_train)
---
```.do_predict(X, clf)```

This function returns both the predicted class and probabilities of the input dataset.

Returns two ```numpy.ndarray```s.

        import firstDown
        y_pred, y_pred_prob = firstDown.train.models.do_predict(X, best_model)
---
```.get_model(model)```

This function returns the selected model.

Returns a ```class```.

        import firstDown
        clf = firstDown.train.models.get_model(model)
---
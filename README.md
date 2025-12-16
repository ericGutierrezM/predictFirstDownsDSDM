# 🏈 _firstDown_: Predicting First Downs in the NFL
#### Computing for Data Science | Barcelona School of Economics | December 2025
> Daniel Campos & Eric Gutiérrez
---
## Motivation
_firstDown_ is a Python library envisioned to get the most out of the vast amount of sports-related data available! Although it has been designed to predict first downs (play success) in the NFL, its structure allows for changes and additions that make possible to perform data analysis on (virtually) any sports. So no matter if it's an extra yard, mile, goal, or point, _firstDown_ will get you there!

## About the dataset: ```nflreadpy```
In the ready-to-implement case provided by _firstDown_, we leverage the large amount of NFL play-by-play and player data from ```nflreadpy```. If you have in mind a sports dataset that you want to test out with _firstDown_, visit the documentation for more information on how to integrate it!

## A note on required libraries
The _pyproject.toml_ file in this repository contains all the dependencies needed for the library to run. Please, make sure to install them all in your system before trying out that model that keeps you up at night!

## On _firstDown_'s structure and scaling
The structure of this library aims at making scaling it easy: enabling the integration of new functionalities and methods. _firstDown_ is organized in sublibraries as following:

> 1. feature_engineering 

- Build new features, and get the most out of them by using encoders.

> 2. graph
- Generate customizable graphs to visualize your analysis.

> 3. hyper_tuning
- Perform a randomized search to get the best hyperparameters for your models.

> 4. load_data
- Load your favorite datasets to perform data analysis and modelling.

> 5. metrics
- Assess the performance of your models.

> 6. preprocessing
- Clean your data and prepare it for analysis.

> 7. train
- Train your models. 

> 8. unit_tests
- Test the methods of the library (ours and the ones you add) using unit testing.
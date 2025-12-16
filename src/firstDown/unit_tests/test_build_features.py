import pandas as pd
import numpy as np
import pytest

from src.firstDown.feature_engineering.build_features import (
    inertia,
    play_type,
    get_positions,
    defense_rush,
    defense_pass,
    defense_scramble
)

def test_inertia_creates_column():
    df = pd.DataFrame({
        'posteam_type': ['home', 'home', 'home', 'away'],
        'first_down': [1, 0, 1, 0]
    })

    result = inertia(df)

    assert 'inertia' in result.columns
    assert len(result['inertia']) == len(df)

def test_inertia_first_row_none():
    df = pd.DataFrame({
        'posteam_type': ['home'],
        'first_down': [1]
    })

    result = inertia(df)

    assert pd.isna(result.loc[0, 'inertia'])

def test_play_type_categories():
    df = pd.DataFrame({
        'qb_scramble': [1.0, 0.0, 0.0, 0.0],
        'play_type': ['pass', 'pass', 'run', 'kick']
    })

    result = play_type(df)

    assert result.loc[0, 'play_category'] == 'scramble'
    assert result.loc[1, 'play_category'] == 'pass'
    assert result.loc[2, 'play_category'] == 'rush'
    assert result.loc[3, 'play_category'] == 'other'

def test_get_positions_merges_correctly():
    pbp = pd.DataFrame({
        'rusher_player_id': ['1'],
        'receiver_player_id': ['2'],
        'passer_player_id': ['3']
    })

    players = pd.DataFrame({
        'gsis_id': ['1', '2', '3'],
        'position': ['RB', 'WR', 'QB']
    })

    result = get_positions(pbp, players)

    assert result.loc[0, 'rush_pos'] == 'RB'
    assert result.loc[0, 'rec_pos'] == 'WR'
    assert result.loc[0, 'pass_pos'] == 'QB'

def test_defense_rush_column_created():
    df = pd.DataFrame({
        'game_id': [1, 2],
        'defteam': ['A', 'A'],
        'game_date': pd.to_datetime(['2021-01-01', '2021-01-08']),
        'play_type': ['run', 'run'],
        'first_down': [1, 0]
    })

    result = defense_rush(df)

    assert 'def_vs_rush' in result.columns

def test_defense_pass_column_created():
    df = pd.DataFrame({
        'game_id': [1, 2],
        'defteam': ['A', 'A'],
        'game_date': pd.to_datetime(['2021-01-01', '2021-01-08']),
        'play_type': ['pass', 'pass'],
        'first_down': [1, 0]
    })

    result = defense_pass(df)

    assert 'def_vs_pass' in result.columns

def test_defense_scramble_column_created():
    df = pd.DataFrame({
        'game_id': [1, 2],
        'defteam': ['A', 'A'],
        'game_date': pd.to_datetime(['2021-01-01', '2021-01-08']),
        'play_type': ['pass', 'pass'],
        'qb_scramble': [1.0, 1.0],
        'first_down': [1, 0]
    })

    result = defense_scramble(df)

    assert 'def_vs_qb_scramble' in result.columns
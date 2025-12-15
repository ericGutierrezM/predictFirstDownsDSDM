import pandas as pd    
import numpy as np

def inertia(dataset):

    inertia_list = []

    for idx in dataset.index:
        idx = idx
        current_team = dataset['posteam_type'][idx]

        past_first_down = []
        past_series_team = []    

        for g in range(1,4):
            try:
                past_first_down.append(dataset['first_down'][idx-g])
                past_series_team.append(dataset['posteam_type'][idx-g])
            except KeyError:
                break
        
        df_series = pd.DataFrame({
            'team': past_series_team,
            'success': past_first_down
        })

        df_series = df_series[df_series['team']==current_team]

        if(len(df_series)==1):
            inertia_list.append(df_series['success'].mean())
        elif(len(df_series)==2):
            inertia_list.append(df_series['success'].iloc[0]*0.7 + df_series['success'].iloc[1]*0.3)
        elif(len(df_series)==3):
            inertia_list.append(df_series['success'].iloc[0]*0.65 + df_series['success'].iloc[1]*0.25 + df_series['success'].iloc[2]*0.1)
        else:
            inertia_list.append(None)    

    dataset['inertia'] = inertia_list

    return dataset

def play_type(dataset):
    dataset['play_category'] =  np.where(dataset['qb_scramble']== 1.0, 'scramble',
                                np.where(dataset['play_type'] == 'pass', 'pass',
                                np.where(dataset['play_type'] == 'run', 'rush',
                                'other')))
    
    return dataset

def get_positions(dataset_pbp, dataset_players):
    data = dataset_pbp.merge(right=dataset_players[['gsis_id','position']], how='left', left_on='rusher_player_id', right_on='gsis_id')
    data = data.drop(['gsis_id','rusher_player_id'], axis=1)
    data = data.rename(columns={"position": "rush_pos"})

    data = data.merge(right=dataset_players[['gsis_id','position']], how='left', left_on='receiver_player_id', right_on='gsis_id')
    data = data.drop(['gsis_id', 'receiver_player_id'], axis=1)
    data = data.rename(columns={"position": "rec_pos"})

    data = data.merge(right=dataset_players[['gsis_id','position']], how='left', left_on='passer_player_id', right_on='gsis_id')
    data = data.drop(['gsis_id', 'passer_player_id'], axis=1)
    data = data.rename(columns={"position": "pass_pos"})

    return data

def defense_rush(dataset):

    results = []
    data_simple = dataset.copy()
    data_simple = data_simple[['game_id', 'defteam', 'game_date', 'play_type', 'first_down']]
    data_simple = data_simple[data_simple['play_type']=='run']

    data_grouped = data_simple.groupby(['game_id','defteam']).agg({'first_down': ['count','sum'], 'game_date': 'first'}).reset_index()
    data_grouped.columns = data_grouped.columns.get_level_values(0)

    for idx in data_grouped.index:
        team = data_grouped.loc[idx, 'defteam']
        date = data_grouped.loc[idx, 'game_date']

        past = data_grouped[ (data_grouped['defteam'] == team) & 
                            (data_grouped['game_date'] < date)]

        past.columns = ['game_id','defteam','total','first_down','game_date']

        if len(past) > 0:
            prop_fd = 1 - (past['first_down'].sum() / past['total'].sum())
        else:
            prop_fd = None

        results.append(prop_fd)
    
    data_grouped['def_vs_rush'] = results

    dataset = dataset.merge(right=data_grouped[['game_id','defteam','def_vs_rush']], how='left', left_on=['game_id','defteam'], right_on=['game_id','defteam'])

    return dataset

def defense_pass(dataset):

    results = []
    data_simple = dataset.copy()
    data_simple = data_simple[['game_id', 'defteam', 'game_date', 'play_type', 'first_down']]
    data_simple = data_simple[data_simple['play_type']=='pass']

    data_grouped = data_simple.groupby(['game_id','defteam']).agg({'first_down': ['count','sum'], 'game_date': 'first'}).reset_index()
    data_grouped.columns = data_grouped.columns.get_level_values(0)

    for idx in data_grouped.index:
        team = data_grouped.loc[idx, 'defteam']
        date = data_grouped.loc[idx, 'game_date']

        past = data_grouped[ (data_grouped['defteam'] == team) & 
                            (data_grouped['game_date'] < date)]

        past.columns = ['game_id','defteam','total','first_down','game_date']

        if len(past) > 0:
            prop_fd = 1 - (past['first_down'].sum() / past['total'].sum())
        else:
            prop_fd = None

        results.append(prop_fd)
    
    data_grouped['def_vs_pass'] = results

    dataset = dataset.merge(right=data_grouped[['game_id','defteam','def_vs_pass']], how='left', left_on=['game_id','defteam'], right_on=['game_id','defteam'])

    return dataset

def defense_scramble(dataset):

    results = []
    data_simple = dataset.copy()
    data_simple = data_simple[['game_id', 'defteam', 'game_date', 'play_type', 'first_down', 'qb_scramble']]
    data_simple = data_simple[data_simple['qb_scramble']==1.0]

    data_grouped = data_simple.groupby(['game_id','defteam']).agg({'first_down': ['count','sum'], 'game_date': 'first'}).reset_index()
    data_grouped.columns = data_grouped.columns.get_level_values(0)

    for idx in data_grouped.index:
        team = data_grouped.loc[idx, 'defteam']
        date = data_grouped.loc[idx, 'game_date']

        past = data_grouped[ (data_grouped['defteam'] == team) & 
                            (data_grouped['game_date'] < date)]

        past.columns = ['game_id','defteam','total','first_down','game_date']

        if len(past) > 0:
            prop_fd = 1 - (past['first_down'].sum() / past['total'].sum())
        else:
            prop_fd = None

        results.append(prop_fd)
    
    data_grouped['def_vs_qb_scramble'] = results

    dataset = dataset.merge(right=data_grouped[['game_id','defteam','def_vs_qb_scramble']], how='left', left_on=['game_id','defteam'], right_on=['game_id','defteam'])

    return dataset

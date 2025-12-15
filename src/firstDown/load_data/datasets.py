import nflreadpy as nfl

def nfl_data():
    pbp = nfl.load_pbp().to_pandas()
    play_by_play = pbp[pbp["season_type"] == "REG"]

    players = nfl.load_players().to_pandas()

    return play_by_play, players

# Easy to add new datasets
import shutil
from pathlib import Path
import random

origin_game_dir = Path('')
difficulty = 'hard'
new_game_dir = Path('')
new_game_num = 50

origin_game_dir = origin_game_dir.joinpath(difficulty)
new_game_dir = new_game_dir.joinpath(difficulty)

for split in origin_game_dir.iterdir():
    if split.name == 'train':
        games = [file.name[:-len(file.suffix)] for file in split.iterdir()]
        games = list(set(games))
        if 'conceptnet_subgraph' in games:
            _ = games.remove('conceptnet_subgraph')
        new_games = random.sample(games, new_game_num)
        for new_game in new_games:
            for suffix in ['.json', '.ni', '.ulx']:
                origin_game_path = origin_game_dir.joinpath(split.name, new_game + suffix)
                new_game_path = new_game_dir.joinpath(split.name, new_game + suffix)
                new_game_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(origin_game_path, new_game_path)
    else:
        for inout in split.iterdir():
            games = [file.name[:-len(file.suffix)] for file in inout.iterdir()]
            games = list(set(games))
            if 'conceptnet_subgraph' in games:
                _ = games.remove('conceptnet_subgraph')
            if len(games) > new_game_num:
                new_games = random.sample(games, new_game_num)
            else:
                new_games = games
            for new_game in new_games:
                for suffix in ['.json', '.ni', '.ulx']:
                    origin_game_path = origin_game_dir.joinpath(split.name, inout.name, new_game + suffix)
                    new_game_path = new_game_dir.joinpath(split.name, inout.name, new_game + suffix)
                    new_game_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(origin_game_path, new_game_path)

print('extracted!')
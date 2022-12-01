from twc_make_game import twc_config, run
from verify_distribution import verify
from pathlib import Path
import shutil
import random
import os

# EasyはTrainの方がTestより多くないければいけない
SAVE_DIR = './games/GAMES_10'
TRAIN_NUM_GAMES = 20
TEST_NUM_GAMES = 16
VALID_NUM_GAMES = 5
GAME_INFO = {
    'hard': [{'objects': 7, 'take': 7, 'rooms': 2, 'ratio': 1.0}], 
    'medium': [{'objects': 3, 'take': 3, 'rooms': 1, 'ratio': 1.0}], 
    'easy': [{'objects': 1, 'take': 1, 'rooms': 1, 'ratio': 1.0}]
}
GAME_SUFFIX = ['.json', '.ni', '.ulx']

config = twc_config()
for difficulty in GAME_INFO.keys():
    print('-'*10 + difficulty + '-'*10)
    train_all_games_num = TRAIN_NUM_GAMES
    in_all_games_num = TEST_NUM_GAMES + VALID_NUM_GAMES
    out_all_games_num = TEST_NUM_GAMES + VALID_NUM_GAMES
    for game_type in GAME_INFO[difficulty]:
        print(f'game_type: {game_type}')
        config.objects = game_type['objects']
        config.rooms = game_type['rooms']
        config.take = game_type['take']
        
        # Train
        config.num_games = int(train_all_games_num * game_type['ratio'])
        config.train, config.test = True, False
        config.output_dir = str(Path(SAVE_DIR).joinpath(difficulty, 'train'))
        print(f'train split num_games:{config.num_games}')
        game_files, train_entities = run(config)
        
        # IN
        config.num_games = int(in_all_games_num * game_type['ratio'])
        config.train, config.test = True, False
        config.output_dir = str(Path(SAVE_DIR).joinpath(difficulty, 'tmp'))
        game_files, _ = run(config, train_entities)
        game_names = [Path(game_file).name[:-4] for game_file in game_files]
        game_names_shuffle = random.sample(game_names, len(game_names))
        in_valid_num, in_test_num = int(config.num_games*(VALID_NUM_GAMES/in_all_games_num)), int(config.num_games*(TEST_NUM_GAMES/in_all_games_num))
        in_valid_games = game_names_shuffle[:in_valid_num]
        in_test_games = game_names_shuffle[in_valid_num:in_valid_num+in_test_num]
        for in_valid_game in in_valid_games:
            for suffix in GAME_SUFFIX:
                origin_path = Path(config.output_dir).joinpath(in_valid_game+suffix)
                move_path = Path(SAVE_DIR).joinpath(difficulty, 'valid', 'in', in_valid_game+suffix)
                move_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(origin_path), str(move_path))
        print('in_valid num_games:', int(len(list(Path(SAVE_DIR).joinpath(difficulty, 'valid', 'in').iterdir())) / 3))
        for in_test_game in in_test_games:
            for suffix in GAME_SUFFIX:
                origin_path = Path(config.output_dir).joinpath(in_test_game+suffix)
                move_path = Path(SAVE_DIR).joinpath(difficulty, 'test', 'in', in_test_game+suffix)
                move_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(origin_path), str(move_path))
        print('in_test num_games:', int(len(list(Path(SAVE_DIR).joinpath(difficulty, 'test', 'in').iterdir())) / 3))
        
        # OUT
        config.num_games = int(out_all_games_num * game_type['ratio'])
        config.train, config.test = False, True
        config.output_dir = str(Path(SAVE_DIR).joinpath(difficulty, 'tmp'))
        print(f'test num_games:{config.num_games}')
        game_files, _ = run(config)
        game_names = [Path(game_file).name[:-4] for game_file in game_files]
        game_names_shuffle = random.sample(game_names, len(game_names))
        out_valid_num, out_test_num = int(config.num_games*(VALID_NUM_GAMES/out_all_games_num)), int(config.num_games*(TEST_NUM_GAMES/out_all_games_num))
        out_valid_games = game_names_shuffle[:out_valid_num]
        out_test_games = game_names_shuffle[out_valid_num:out_valid_num+out_test_num]
        for out_valid_game in out_valid_games:
            for suffix in GAME_SUFFIX:
                origin_path = Path(config.output_dir).joinpath(out_valid_game+suffix)
                move_path = Path(SAVE_DIR).joinpath(difficulty, 'valid', 'out', out_valid_game+suffix)
                move_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(origin_path), str(move_path))
        print('out_valid num_games:', int(len(list(Path(SAVE_DIR).joinpath(difficulty, 'valid', 'out').iterdir())) / 3))
        for out_test_game in out_test_games:
            for suffix in GAME_SUFFIX:
                origin_path = Path(config.output_dir).joinpath(out_test_game+suffix)
                move_path = Path(SAVE_DIR).joinpath(difficulty, 'test', 'out', out_test_game+suffix)
                move_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(origin_path), str(move_path))
        print('out_test num_games:', int(len(list(Path(SAVE_DIR).joinpath(difficulty, 'test', 'out').iterdir())) / 3))
        os.rmdir(str(Path(SAVE_DIR).joinpath(difficulty, 'tmp')))
        
verify(SAVE_DIR)
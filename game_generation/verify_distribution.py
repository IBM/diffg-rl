from cgi import test
from pathlib import Path
import json

def verify(game_dir, generate_entities=False):
    game_dir = Path(game_dir)
    
    all_entities = []

    for difficulty in game_dir.iterdir():
        if difficulty.name == 'tmp':
            continue
        print('=' * 20)
        print(difficulty.name)
        print('-' * 20)
        if not difficulty.is_dir():
            continue
        train_item_entities, in_item_entities, out_item_entities = [], [], []
        train_entities, in_entities, out_entities = [], [], []
        train_sets, in_sets, out_sets = [], [], []
        for split in difficulty.iterdir():
            print('-'*3 + str(split.name) + '-'*3)
            if split.name == 'train':
                print(int(len(list(split.iterdir()))/3))
                for file in split.iterdir():
                    if file.suffix == '.json':
                        with open(file, 'r') as f:
                            d = json.load(f)
                        train_item_entities.extend(list(d['metadata']['goal_locations'].keys()))
                        train_entities.extend(list(d['metadata']['goal_locations'].keys()))
                        for v in d['metadata']['goal_locations'].values():
                            train_entities.extend(v)
                        if list(d['metadata']['goal_locations'].keys()) in train_sets:
                            print(file.name)
                            print(list(d['metadata']['goal_locations'].keys()))
                        else:
                            train_sets.append(list(d['metadata']['goal_locations'].keys()))
            else:
                for inout in split.iterdir():
                    print('-'*3 + str(inout.name) + '-'*3)
                    print(int(len(list(inout.iterdir()))/3))
                    if inout.name == 'in':
                        for file in inout.iterdir():
                            if file.suffix == '.json':
                                with open(file, 'r') as f:
                                    d = json.load(f)
                                in_item_entities.extend(list(d['metadata']['goal_locations'].keys()))
                                in_entities.extend(list(d['metadata']['goal_locations'].keys()))
                                for v in d['metadata']['goal_locations'].values():
                                    in_entities.extend(v)
                                if list(d['metadata']['goal_locations'].keys()) in in_sets:
                                    print(file.name)
                                    print(list(d['metadata']['goal_locations'].keys()))
                                else:
                                    in_sets.append(list(d['metadata']['goal_locations'].keys()))
                    elif inout.name == 'out':
                        for file in inout.iterdir():
                            if file.suffix == '.json':
                                with open(file, 'r') as f:
                                    d = json.load(f)
                                out_item_entities.extend(list(d['metadata']['goal_locations'].keys()))
                                out_entities.extend(list(d['metadata']['goal_locations'].keys()))
                                for v in d['metadata']['goal_locations'].values():
                                    out_entities.extend(v)
                                if list(d['metadata']['goal_locations'].keys()) in out_sets:
                                    print(file.name)
                                    print(list(d['metadata']['goal_locations'].keys()))
                                else:
                                    out_sets.append(list(d['metadata']['goal_locations'].keys()))

        print('-' * 5)
        print('trainsets:', len(train_sets))
        print('insets:', len(in_sets))
        print('outsets:', len(out_sets))
        
        train_item_entities = set(train_item_entities)
        in_item_entities = set(in_item_entities)
        out_item_entities = set(out_item_entities)

        print('train_item_entities:', len(train_item_entities))
        print('in_item_entities:', len(in_item_entities))
        print('out_item_entities:', len(out_item_entities))
        print('entities in IN set but out of train set:', len(in_item_entities - train_item_entities))
        print('entities both in OUT set and train set:', len(out_item_entities & train_item_entities))
        
        all_entities.extend(train_entities)
        all_entities.extend(in_entities)
        all_entities.extend(out_entities)
    
    if generate_entities:
        all_entities = list(set(all_entities))
        print('-'*5)
        print('all_entities:', len(all_entities))
        with open(game_dir.joinpath('entities.txt'), 'w') as f:
            f.write('\n'.join(all_entities))
        print('Saved entities.txt')
    

if __name__ == "__main__":
    game_dir = './games/GAMES_100'
    verify(game_dir, generate_entities=True)
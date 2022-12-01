from operator import truediv
from pathlib import Path
import json
import multiprocessing
import pickle

import networkx as nx

from utils.kg import (construct_kg_vg, similar_entity_context_subgraph, similar_entity_subgraph, shortest_path_subgraph)


DIFFICULTY = "hard"
THRESHOLD = 0.5
base = False
alias = True
nbc = True
mode_str = f"DIFFICULTY_{DIFFICULTY}_BASE_{base}_TH_{THRESHOLD}_nbc_{nbc}_alias_{alias}"
print(f"(Mode) {mode_str}")
Path(f"./calc_pr/results/{mode_str}").mkdir(exist_ok=True, parents=True)

vg_graph, _, _ = construct_kg_vg('./vg/relationships.json')
print("")
with open(f"./similar_dicts/similarity_entity_VG_add_{THRESHOLD}_base.json", "r") as f:
    similar_dict = json.load(f)

BASE_DIR = "./games/GAMES_100"
test_game_json_dir = Path(f"{BASE_DIR}/{DIFFICULTY}/test/out")
valid_game_json_dir = Path(f"{BASE_DIR}/{DIFFICULTY}/valid/out")

game_json_list = [path for path in test_game_json_dir.iterdir() if path.suffix == ".json"] + [path for path in valid_game_json_dir.iterdir() if path.suffix == ".json"]

def calc_pr_rc(game_json):
    with open(game_json, "r") as f:
        game_info = json.load(f)
    entities = game_info["metadata"]["entities"]
    goal_locations = game_info["metadata"]["goal_locations"]
    object_entities = list(goal_locations.keys())
    container_entities = list(set(entities) - set(object_entities))
    world_graph = nx.DiGraph()

    if base:
        world_graph = shortest_path_subgraph(vg_graph, world_graph, entities)
    else:
        if nbc:
            world_graph = similar_entity_context_subgraph(vg_graph, world_graph, object_entities, container_entities, similar_dict, alias=alias)
        else:
            world_graph = similar_entity_subgraph(vg_graph, world_graph, entities, similar_dict, alias=alias)

    recall_n = 0
    precision_n = []
    for object, locations in goal_locations.items():
        if alias:
            object_list = [object]
            location_list = locations
        else:
            is_in_similar_dict = True
            if object not in similar_dict.keys():
                is_in_similar_dict = False
            for location in locations:
                if location not in similar_dict.keys():
                    is_in_similar_dict = False
            if not is_in_similar_dict:
                continue

            is_counted = False
            object_list = [rel_o[0] for rel_o in similar_dict[object]]
            location_list = [rel_l[0] for loc in locations for rel_l in similar_dict[loc]]
            location_list = list(set(location_list))

        for (o, l) in list(world_graph.edges):
            if alias:
                if o in object_list[0] and l in location_list:
                    recall_n += 1
                    break
            else:
                if o in object_list and l in location_list:
                    precision_n.append((o, l))
                    if not is_counted:
                        recall_n += 1
                        is_counted = True

    game_id = game_json.name.split(".")[-2].split("-")[-1]
    print(f"game_id: {game_id}, PrecisionN: {len(precision_n)} , Recall: ({recall_n}/{len(goal_locations)})")
    
    with open(Path("./calc_pr/results/").joinpath(mode_str, game_id + '.pkl'), 'wb') as f:
        pickle.dump({"recall_d": len(goal_locations), "recall_n": recall_n, "precidion_d": list(world_graph.edges), "precision_n": precision_n}, f)

    return list(world_graph.edges), precision_n, len(goal_locations), recall_n

    # all_precision_d.extend(list(world_graph.edges))
    # all_precision_n.extend(precision_n)
    # all_recall_n += recall_n
    # all_recall_d += len(goal_locations)

    # for object, locations in goal_locations.items():
    #     if object in all_goal_locations.keys():
    #         all_goal_locations[object] = list(set(all_goal_locations[object]) | set(locations))
    #     else:
    #         all_goal_locations[object] = locations

all_precision_d = []
all_precision_n = []
all_recall_d = 0
all_recall_n = 0

with multiprocessing.Pool(8) as pool:
    results = pool.map(calc_pr_rc, game_json_list)

for result in results:
    all_precision_d.extend(result[0])
    all_precision_n.extend(result[1])
    all_recall_d += result[2]
    all_recall_n += result[3]

if alias:
    all_precision_n = all_recall_n
else:
    all_precision_n = len(all_precision_n)

all_precision_d = len((all_precision_d)) # setにするかどうか

print(f"Precision: {all_precision_n}/{all_precision_d}({all_precision_n/all_precision_d*100:,.1f}%)")
print(f"Recall: {all_recall_n}/{all_recall_d}({all_recall_n/all_recall_d*100:,.1f}%)")

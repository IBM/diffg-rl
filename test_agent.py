import csv
import json
import os
import pickle
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from logging import (CRITICAL, DEBUG, ERROR, INFO, WARNING, FileHandler,
                     Formatter, StreamHandler, getLogger)
from time import time

import numpy as np
import torch

import agent
from config import model_config
from games import dataset
from train_agent import episode
from utils import extractor
from utils.generic import getUniqueFileHandler
from utils.kg import (RelationExtractor, construct_kg, construct_kg_vg,
                      load_manual_graphs)
from utils.nlp import Tokenizer
from utils.textworld_utils import get_goal_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(agent, opt, logger, random_action=False):
    filter_examine_cmd = False
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    manual_world_graphs = {}
    if opt.dataset_type == 'old':
        game_path = opt.game_dir + "/" + (
            str(opt.difficulty_level) + "/" + opt.mode  if opt.difficulty_level != '' else opt.game_dir + "/" + opt.mode )
        if opt.graph_emb_type and 'world' in opt.graph_type:
            print("Loading Manual World Graphs ... ", end='')
            manual_world_graphs = load_manual_graphs(game_path + '/manual_subgraph_brief')
    elif opt.dataset_type == 'new':
        game_path = os.path.join(opt.game_dir, opt.difficulty_level, 'test', opt.mode)
    else:
        raise ValueError('This mode is not implemented!')
    
    if opt.graph_emb_type and 'world' in opt.graph_type:
        print("Loading Knowledge Graph ... ", end='')
        if opt.world_source_type == 'VG':
            agent.kg_graph, _, _= construct_kg_vg('./vg/relationships.json')
        elif opt.world_source_type == 'CN':
            agent.kg_graph, _, _= construct_kg(game_path + '/conceptnet_subgraph.txt')
        else:
            raise ValueError('This mode is not implemented!')
        print(' DONE')
        # optional: Use complete or brief manually extracted conceptnet subgraph for the agent
    
    if opt.graph_emb_type and 'world' in opt.graph_type and opt.world_evolve_type in ['EbM', 'EbMNbC']:
        logger.info(f'Similar Dict: {os.path.basename(opt.similar_dict_path)}')
        with open(opt.similar_dict_path, 'r') as f:
            agent.similar_dict = json.load(f)

    if opt.game_name:
        game_path = game_path + "/"+ opt.game_name

    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, opt.max_step_per_episode, opt.batch_size,
                                                opt.mode, opt.verbose)
    # Get Goals as graphs
    goal_graphs = {}
    for game_file in env.gamefiles:
        goal_graph = get_goal_graph(game_file)
        if goal_graph:
            game_id = game_file.split('-')[-1].split('.')[0]
            goal_graphs[game_id] = goal_graph

    # Collect some statistics: nb_steps, final reward.
    total_games_count = len(game_file_names)
    game_identifiers, avg_moves, avg_scores, avg_norm_scores, max_poss_scores = [], [], [], [], []

    success_commands = []
    if opt.graph_extract_mode:
        exp = {}
    for no_episode in (range(opt.nepisodes)):
        logger.debug('='*200)
        logger.debug(f'episode: {no_episode}')
        if not random_action:
            random.seed(no_episode)
            np.random.seed(no_episode)
            torch.manual_seed(no_episode)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(no_episode)
            env.seed(no_episode)

        agent.start_episode(opt.batch_size)
        avg_eps_moves, avg_eps_scores, avg_eps_norm_scores = [], [], []
        num_games = total_games_count
        game_max_scores = []
        game_names = []
        # success_records = []
        # failure_records = []
        while num_games > 0:
            # records = []
            # if opt.agent_type == 'knowledgeaware':
            #     record_graphs = {'local': [], 'world': []}
            obs, infos = env.reset()  # Start new episode.
            if filter_examine_cmd:
                for commands_ in infos["admissible_commands"]: # [open refri, take apple from refrigeration]
                    for cmd_ in [cmd for cmd in commands_ if cmd.split()[0] in ["examine", "look"]]:
                        commands_.remove(cmd_)

            batch_size = len(obs)
            num_games -= len(obs)
            game_goal_graphs = [None] * batch_size
            max_scores = []
            game_ids = []
            game_manual_world_graph = [None] * batch_size
            for b, game in enumerate(infos["game"]):
                max_scores.append(game.max_score)
                if "uuid" in game.metadata:
                    game_id = game.metadata["uuid"].split("-")[-1]
                    game_ids.append(game_id)
                    game_names.append(game_id)
                    game_max_scores.append(game.max_score)
                    if len(goal_graphs):
                        game_goal_graphs[b] = goal_graphs[game_id]
                    if len(manual_world_graphs):
                        game_manual_world_graph[b] = manual_world_graphs[game_id]

            if not game_ids:
                game_ids = range(num_games,num_games+batch_size)
                game_names.extend(game_ids)

            commands = ["restart"]*len(obs)
            scored_commands = [[] for b in range(batch_size)]
            last_scores = [0.0]*len(obs)
            scores = [0.0]*len(obs)
            dones = [False]*len(obs)
            nb_moves = [0]*len(obs)
            infos["goal_graph"] = game_goal_graphs
            infos["manual_world_graph"] = game_manual_world_graph
            infos["game_difficulty"] = opt.difficulty_level
            agent.reset_parameters(opt.batch_size)
            # records.append({"step": "init", "game_id": game_ids, "goal_graph": infos['goal_graph'][0].edges, "max_score": infos['max_score'], "manual": infos['manual_world_graph'], "entities": infos['entities'], "admissible_commands": infos['admissible_commands'], "inventory": infos['inventory'], "description": infos['description']})
            logger.debug(f"step: init\ngoal_graph: {infos['goal_graph'][0].edges}\nmax_score: {infos['max_score']}\nmanual: {infos['manual_world_graph']}\nentities: {infos['entities']}\nadmissible_commands: {infos['admissible_commands']}\ninventory: {infos['inventory']}\ndescription: {infos['description']}")
            for step_no in range(opt.max_step_per_episode):
                nb_moves = [step + int(not done) for step, done in zip(nb_moves, dones)]

                if agent.graph_emb_type and ('local' in agent.graph_type or 'world' in agent.graph_type):
                    agent.update_current_graph(obs, commands, scored_commands, infos, opt.graph_mode, step_no=step_no, goal_world=opt.goal_world)
                
                if opt.graph_extract_mode:
                    exp[game_ids[0]] = {'goal_graph': list(infos['goal_graph'][0].edges), 'object_entities': agent.object_entities[0], 'container_entities': agent.container_entities[0], 'world_graph_nodes': list(agent.world_graph[0].nodes), 'world_graph_edges': list(agent.world_graph[0].edges)}
                    break

                commands = agent.act(obs, scores, dones, infos, scored_commands, random_action, opt.truncate)
                obs, scores, dones, infos = env.step(commands)
                infos["goal_graph"] = game_goal_graphs
                infos["manual_world_graph"] = game_manual_world_graph
                infos["game_difficulty"] = [opt.difficulty_level] * batch_size
                if 'local' in opt.graph_type:
                    logger.debug(f'local graph: {dict(agent.local_graph[0].edges)}')
                if 'world' in opt.graph_type:
                    logger.debug(f'world graph: {dict(agent.world_graph[0].edges)}')
                logger.debug('-'*100)
                logger.debug(f"step: {step_no}\ncommands: {commands}\nobs: {obs}\nscores: {scores}\nobject_entities: {agent.object_entities[0]}\ncontainer_entities: {agent.container_entities[0]}")
                # records.append({"step": step_no, "dones": dones, "scores": scores, "obs": obs, "command": commands, "inventory": infos['inventory'], "entities": infos['entities'], "admissible_commands": infos['admissible_commands'], "last_action": infos['last_action'], "facts": infos['facts'], 'object_entities': agent.object_entities[0], 'container_entities': agent.container_entities[0]})
                
                # if opt.agent_type == 'knowledgeaware':
                    # if 'local' in opt.graph_type:
                    #     record_graphs['local'].append(agent.local_graph[0])
                    # if 'world' in opt.graph_type:
                    #     record_graphs['world'].append(agent.world_graph[0])

                for b in range(batch_size):
                    if scores[b] - last_scores[b] > 0:
                        last_scores[b] = scores[b]
                        scored_commands[b].append(commands[b])
                        success_commands.append(commands[b])

                if all(dones):
                    break
                if step_no == opt.max_step_per_episode - 1:
                    dones = [True for _ in dones]
            
            if opt.graph_extract_mode:
                print(".", end="")
                continue
            
            agent.act(obs, scores, dones, infos, scored_commands, random_action, opt.truncate)  # Let the agent know the game is done.

            avg_eps_moves.extend(nb_moves)
            avg_eps_scores.extend(scores)
            avg_eps_norm_scores.extend([score/max_score for score, max_score in zip(scores, max_scores)])
            logger.debug('=' * 200)
            # records_graphs = {'records': records, 'graphs': record_graphs}
            # log_record(records_graphs, logger, opt.graph_type)
            if opt.verbose:
                print(".", end="")
            # if min([score/max_score for score, max_score in zip(scores, max_scores)]) != 1.0:
            #     failure_records.append({'records': records, 'graphs': record_graphs})
            # else:
            #     success_records.append({'records': records, 'graphs': record_graphs})
        
        if opt.graph_extract_mode:
            break
        # logger.debug('=' * 200)
        # logger.debug('success')
        # for success in success_records:
        #     logger.debug('=' * 200)
        #     log_record(success, logger, opt.graph_type)
        # logger.debug('=' * 200)
        # logger.debug('failure')
        # for failure in failure_records:
        #     logger.debug('=' * 200)
        #     log_record(failure, logger, opt.graph_type)
        logger.debug(f'{no_episode} episode validation, norm score: {np.mean(avg_eps_norm_scores)}, steps: {np.mean(avg_eps_moves)}')
        if opt.verbose:
            print("*", end="")
        agent.end_episode()
        game_identifiers.append(game_names)
        avg_moves.append(avg_eps_moves) # episode x # games
        avg_scores.append(avg_eps_scores)
        avg_norm_scores.append(avg_eps_norm_scores)
        max_poss_scores.append(game_max_scores)
    if opt.graph_extract_mode:
        graph_dir = f'./logs/analyze/graphs/test'
        graph_path = os.path.join(graph_dir, f'{opt.exp_name}.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(exp, f)
        logger.info('graph extracted!')
        exit()
    logger.info("=" * 200)
    logger.info('Success Commands')
    logger.info(Counter(success_commands).most_common())
    env.close()
    game_identifiers = np.array(game_identifiers)
    avg_moves = np.array(avg_moves)
    avg_scores = np.array(avg_scores)
    avg_norm_scores = np.array(avg_norm_scores)
    max_poss_scores = np.array(max_poss_scores)
    if opt.verbose:
        idx = np.apply_along_axis(np.argsort, axis=1, arr=game_identifiers)
        game_avg_moves = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_moves))), axis=0)
        game_norm_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_norm_scores))), axis=0)
        game_avg_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_scores))), axis=0)

        msg = "\nGame Stats:\n-----------\n" + "\n".join(
            "  Game_#{}: {} = Score: {:5.2f} Norm_Score: {:5.2f} Moves: {:5.2f}/{}".format(game_no, game_name, avg_score,
                                                                                            norm_score, avg_move,
                                                                                            opt.max_step_per_episode)
            for game_no, (game_name, norm_score, avg_score, avg_move) in
            enumerate(zip(game_names, game_norm_scores, game_avg_scores, game_avg_moves)))

        logger.info(msg)

        total_avg_moves = np.mean(game_avg_moves)
        total_avg_scores = np.mean(game_avg_scores)
        total_norm_scores = np.mean(game_norm_scores)
        msg = opt.mode+" stats: avg. score: {:4.2f}; norm. avg. score: {:4.2f}; \n"
        logger.info(msg.format(total_avg_scores, total_norm_scores))

        ## Dump log files ......
        str_result = {opt.mode + 'game_ids': game_identifiers, opt.mode + 'max_scores': max_poss_scores,
                      opt.mode + 'scores_runs': avg_scores, opt.mode + 'norm_score_runs': avg_norm_scores,
                      opt.mode + 'moves_runs': avg_moves}
        
        episode_avg_moves = np.mean(avg_moves, axis=1)
        episode_avg_norm_scores = np.mean(avg_norm_scores, axis=1)
        
        episode_avg_moves = episode_avg_moves.tolist()
        episode_avg_norm_scores = episode_avg_norm_scores.tolist()
                
        with open(opt.results_filename, 'wb') as f:
            pickle.dump(str_result, f)
    return total_norm_scores, total_avg_moves


def log_record(record_dict, logger, graph_type):
    records, record_graphs = record_dict['records'], record_dict['graphs']
    for r in range(len(records)):
        logger.debug('-' * 100)
        for k, v in records[r].items():
            if type(v) is not str:
                v = str(v)
            logger.debug(k + ': ' + v)
        if r == 0:
            continue
        if opt.agent_type == 'knowledgeaware':
            for graph in graph_type:
                logger.debug(f'{graph} graph nodes: {dict(record_graphs[graph][r-1].nodes)}')
                logger.debug(f'{graph} graph edges: {dict(record_graphs[graph][r-1].edges)}')
                

def check_similar_entities(entities, similar_dict):
    check_dict = defaultdict(list)
    for entity in entities:
        if entity in similar_dict.keys():
            for similar_entity in similar_dict[entity]:
                check_dict[entity].append(similar_entity[0])
    return check_dict


def init_logger(log_path, modname=__name__):
    logger = getLogger('log')
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(INFO)
    sh_formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(sh_formatter)
    logger.addHandler(sh)
    
    fh = FileHandler(log_path)
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    return logger


if __name__ == '__main__':
    
    opt = model_config()
    
    random.seed(opt.initial_seed)
    np.random.seed(opt.initial_seed)
    torch.manual_seed(opt.initial_seed)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.initial_seed)
        torch.backends.cudnn.deterministic = True
    opt.seed = opt.initial_seed
    # yappi.start()

    scores_runs = []
    norm_score_runs = []
    moves_runs = []
    test_scores_runs = []
    test_norm_score_runs = []
    test_moves_runs = []

    random_action = False
    if opt.agent_type == 'random':
        random_action = True
        opt.graph_emb_type = None
    if opt.agent_type == 'simple':
        opt.graph_type = ''
        opt.graph_emb_type = None
        
    if opt.auto_pretrained_model:
        models = os.listdir(opt.pretrained_model)
        if 'episode_last.pt' in models:
            models.remove('episode_last.pt')
        if 'interval' in models:
            models.remove('interval')
        models.sort(key=lambda x: int(re.search(r'[0-9]+', x).group()), reverse=True)
        opt.pretrained_model = os.path.join(opt.pretrained_model, models[0])
        model_num = re.search(r'[0-9]+', models[0]).group()
        opt.elements = f'{opt.elements}_TESTEPISODE_{model_num}'

    tk_extractor = extractor.get_extractor(opt.token_extractor)

    base_dir = f"{opt.logs_dir}/test/{opt.exp_name}/{os.path.basename(os.path.dirname(opt.pretrained_model))}/"
    os.makedirs(base_dir, exist_ok=True)
    csv_filename = base_dir + 'results.csv'
    
    logs_dir = base_dir + "log/"
    os.makedirs(logs_dir, exist_ok=True)
    logger = init_logger(os.path.join(logs_dir, f'{opt.elements}.log'))
    
    if opt.world_evolve_type == 'EbMNbC':
        opt.world_graph_cache = os.path.join(base_dir, 'world_graph_cache')
        os.makedirs(opt.world_graph_cache, exist_ok=True)
    
    results_filename = base_dir + opt.results_dir + '/'
    os.makedirs(results_filename, exist_ok=True)
    opt.results_filename = results_filename + f"{opt.elements}.pkl"
    
    opt.amr_save_cache = os.path.join(opt.amr_save_cache, 'test', opt.exp_name)
    opt.amr_load_cache = os.path.join(opt.amr_load_cache, 'test_base/amr_cache.pkl')
    if os.path.exists(opt.amr_load_cache):
        os.makedirs(opt.amr_save_cache, exist_ok=True)
        shutil.copyfile(opt.amr_load_cache, os.path.join(opt.amr_save_cache, 'base_amr_cache.pkl'))
        
    if opt.world_evolve_type in ['EbM', 'EbMNbC']:
        similar_dict_cache = os.path.join(opt.amr_save_cache, os.path.basename(opt.similar_dict_path))
        shutil.copyfile(opt.similar_dict_path, similar_dict_cache)
        opt.similar_dict_path = similar_dict_cache

    logger.info(opt.exp_name)
    logger.info(opt)
    
    graph = None
    for n in range(0, opt.nruns):
        # opt.run_no = n

        print("Testing ...")
        emb_types = [opt.word_emb_type, opt.graph_emb_type]
        tokenizer = Tokenizer(noun_only_tokens=opt.noun_only_tokens, use_stopword=opt.use_stopword, ngram=opt.ngram,
                              extractor=tk_extractor)
        rel_extractor = RelationExtractor(tokenizer, openie_url=opt.corenlp_url, amr_server_ip=opt.amr_server_ip, amr_server_port=opt.amr_server_port, amr_save_cache=opt.amr_save_cache, amr_load_cache=opt.amr_load_cache)
        if opt.dropout_ratio == 0.0:
            opt.dropout_ratio = None
        myagent = agent.KnowledgeAwareAgent(graph, opt, tokenizer, rel_extractor, device)
        myagent.type = opt.agent_type

        print('Loading Pretrained Model ...',end='')
        if torch.cuda.is_available():
            checkpoint = torch.load(opt.pretrained_model)
        else:
            checkpoint = torch.load(opt.pretrained_model, map_location=torch.device('cpu'))
        myagent.model.load_state_dict(checkpoint['model_state_dict'])
        # myagent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('DONE')

        myagent.test(opt.batch_size)
        opt.mode = opt.split
        opt.nepisodes=opt.no_eval_episodes # for testing
        opt.max_step_per_episode = opt.eval_max_step_per_episode
        starttime = time()
        logger.info('=' * 200)
        logger.info('-' * 90 + f' RUN: {n}, SEED: {opt.seed} ' + '-' * 90)
        logger.info('=' * 200)
        total_norm_score, total_avg_moves = play(myagent, opt, logger, random_action=random_action)
        if os.path.exists(csv_filename):
            with open(csv_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([opt.seed, total_norm_score, total_avg_moves])
        else:
            with open(csv_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['seed', 'norm_score', 'steps'])
                writer.writerow([opt.seed, total_norm_score, total_avg_moves])
        process_time = time() - starttime
        logger.info("Tested in {} mins {:.2f} secs".format(process_time // 60, process_time % 60))

    if not os.path.exists(os.path.join(opt.amr_save_cache, 'amr_cache.pkl')):
        shutil.rmtree(opt.amr_save_cache)

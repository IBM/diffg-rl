import numpy as np
import random
from time import time
import torch
import pickle
import shutil
import agent
from config import model_config
from utils import extractor
from utils.generic import getUniqueFileHandler
from utils.kg import construct_kg, construct_kg_vg, load_manual_graphs, RelationExtractor
from utils.textworld_utils import get_goal_graph
from utils.nlp import Tokenizer
from games import dataset
import os
import json
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL
from tensorboardX import SummaryWriter
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def episode(no_episode, opt, env, agent, total_games_count, goal_graphs, manual_world_graphs, logger, random_action=False, filter_examine_cmd=False):
    if not random_action:
        random.seed(no_episode)
        np.random.seed(no_episode)
        torch.manual_seed(no_episode)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(no_episode)
        env.seed(no_episode)

    agent.start_episode(opt.batch_size)
    step_eps_count = 0
    reset_eps_count = 0
    avg_eps_moves, avg_eps_scores, avg_eps_norm_scores = [], [], []
    num_games = total_games_count
    game_max_scores = []
    game_names = []
    if opt.graph_extract_mode:
        exp = {}
    while num_games > 0:
        obs, infos = env.reset()  # Start new episode.
        reset_eps_count += 1
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
        infos["game_difficulty"] = [opt.difficulty_level] * batch_size
        agent.reset_parameters(opt.batch_size)
        if opt.train_log:
            logger.debug('=' * 100)
            if agent.mode == 'train':
                logger.debug('=' * 40 + f' episode: {no_episode}, training, game_id: {game_ids} ' + '=' * 40)
            else:
                logger.debug('=' * 40 + f' episode: {no_episode}, validation, game_id: {game_ids} ' + '=' * 40)
            logger.debug('=' * 100)
            logger.debug(f"step: init\ngoal_graph: {infos['goal_graph'][0].edges}\nmax_score: {infos['max_score']}\nmanual: {infos['manual_world_graph']}\nentities: {infos['entities']}\nadmissible_commands: {infos['admissible_commands']}\ninventory: {infos['inventory']}\ndescription: {infos['description']}")
        is_skip = False
        for step_no in range(opt.max_step_per_episode):
            nb_moves = [step + int(not done) for step, done in zip(nb_moves, dones)]

            if agent.graph_emb_type and ('local' in agent.graph_type or 'world' in agent.graph_type):
                # prune_nodes = opt.prune_nodes if no_episode >= opt.prune_episode and no_episode % 25 ==0 and step_no % 10 == 0 else False
                prune_nodes = opt.prune_nodes
                agent.update_current_graph(obs, commands, scored_commands, infos, opt.graph_mode, prune_nodes=prune_nodes, step_no=step_no, goal_world=opt.goal_world)
                
            if opt.graph_extract_mode:
                exp[game_ids[0]] = {'goal_graph': list(infos['goal_graph'][0].edges), 'object_entities': agent.object_entities[0], 'container_entities': agent.container_entities[0], 'world_graph_nodes': list(agent.world_graph[0].nodes), 'world_graph_edges': list(agent.world_graph[0].edges)}
                break

            commands = agent.act(obs, scores, dones, infos, scored_commands, random_action, opt.truncate, no_episode=no_episode)
            obs, scores, dones, infos = env.step(commands)
            step_eps_count += 1
            infos["goal_graph"] = game_goal_graphs
            infos["manual_world_graph"] = game_manual_world_graph
            infos["game_difficulty"] = [opt.difficulty_level] * batch_size
            if opt.train_log:
                logger.debug(f"step: {step_no}\ncommands: {commands}\nobs: {obs}\nscores: {scores}")
                # if agent.mode == 'train':
                #     trainsitions = [l[0] for l in agent.transitions[0]]
                #     if len(trainsitions) >= 2:
                #         logger.debug(f"rewards: {trainsitions[-2]}")
                #     for k, v in agent.batch_stats[0]['max'].items():
                #         logger.debug(f'{k}: {v}')
                #     for k, v in agent.batch_stats[0]['mean'].items():
                #         logger.debug(f'{k}: {v}')
                # if 'local' in opt.graph_type:
                #     logger.debug(f'local graph: {dict(agent.local_graph[0].edges)}')
                # if 'world' in opt.graph_type:
                #     logger.debug(f'world graph: {dict(agent.world_graph[0].edges)}')

            for b in range(batch_size):
                if scores[b] - last_scores[b] > 0:
                    last_scores[b] = scores[b]
                    scored_commands[b].append(commands[b])

            if all(dones):
                break
            if step_no == opt.max_step_per_episode - 1:
                dones = [True for _ in dones]
                
        if is_skip:
            continue
        
        if opt.graph_extract_mode:
            print(".", end="")
            continue
        
        agent.act(obs, scores, dones, infos, scored_commands, random_action, opt.truncate)  # Let the agent know the game is done.

        if opt.verbose:
            print(".", end="")
        avg_eps_moves.extend(nb_moves)
        avg_eps_scores.extend(scores)
        avg_eps_norm_scores.extend([score/max_score for score, max_score in zip(scores, max_scores)])
    
    if opt.graph_extract_mode:
        graph_dir = f'./logs/analyze/graphs/train'
        graph_path = os.path.join(graph_dir, f'{opt.exp_name}.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(exp, f)
        logger.info('graph extracted!')
        exit()
    
    agent.end_episode()
    
    return game_names, avg_eps_moves, avg_eps_scores, avg_eps_norm_scores, game_max_scores, reset_eps_count, step_eps_count


def play(agent, opt, logger, random_action=False, writer=None):
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    manual_world_graphs = {}
    if opt.dataset_type == 'old':
        game_path = opt.game_dir + "/" + (
            str(opt.difficulty_level) + "/" + opt.mode  if opt.difficulty_level != '' else opt.game_dir + "/" + opt.mode )
        if opt.graph_emb_type and 'world' in opt.graph_type:
            print("Loading Manual World Graphs ... ", end='')
            manual_world_graphs = load_manual_graphs(game_path + '/' + opt.manual_graph)
    elif opt.dataset_type == 'new':
        game_path = os.path.join(opt.game_dir, opt.difficulty_level, opt.mode)
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
    reset_counts, setup_counts = [], []
    
    # if opt.valid != 'none':
    #     valid_game_path = opt.game_dir + "/" + (
    #     str(opt.difficulty_level) + "/" + opt.valid  if opt.difficulty_level != '' else opt.game_dir + "/" + opt.valid )

    if opt.valid != 'none':
        validation_scores = []
        save_episodes = []
        max_mean_validation_score = 0
        last_save_episode = 0
        max_validation_score = 0
        min_validation_step = 100
        model_parameters = {}
        optimizer_parameters = {}
        
    if opt.pretrained_model != '':
        print('Loading Pretrained Model ...',end='')
        if torch.cuda.is_available():
            checkpoint = torch.load(opt.pretrained_model)
        else:
            checkpoint = torch.load(opt.pretrained_model, map_location=torch.device('cpu'))
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cache_point = opt.pretrained_model.replace('checkpoints', 'cache')
        cache = torch.load(cache_point)
        validation_scores, save_episodes, max_mean_validation_score, last_save_episode, max_validation_score, min_validation_step, model_parameters, optimizer_parameters = cache['validation_scores'], cache['save_episodes'], cache['max_mean_validation_score'], cache['last_save_episode'], cache['max_validation_score'], cache['min_validation_step'], cache['model_parameters'], cache['optimizer_parameters']
        print('DONE')
    
    for no_episode in (range(opt.start_ep, opt.start_ep+opt.nepisodes)):
        game_names, avg_eps_moves, avg_eps_scores, avg_eps_norm_scores, game_max_scores, reset_eps_count, step_eps_count = episode(no_episode, opt, env, agent, total_games_count, goal_graphs, manual_world_graphs, logger, random_action=random_action)
        
        if opt.graph_extract_mode:
            break
        if opt.verbose:
            print("*", end="")
        game_identifiers.append(game_names)
        avg_moves.append(avg_eps_moves) # episode x # games
        avg_scores.append(avg_eps_scores)
        avg_norm_scores.append(avg_eps_norm_scores)
        max_poss_scores.append(game_max_scores)
        reset_counts.append(reset_eps_count)
        setup_counts.append(step_eps_count)

        writer.add_scalar('train/norm_scores', np.mean(avg_eps_norm_scores), no_episode)
        writer.add_scalar('train/steps', np.mean(avg_eps_moves), no_episode)
        
        # Validation
        if opt.valid != 'none' and no_episode % opt.valid_interval == 0:
            if opt.valid == 'both':
                splits = ['valid', 'test']
            else:
                splits = [opt.valid]
            
            is_early_stopping = False
            valid_manual_world_graphs = {}
            for split in splits:
                if opt.dataset_type == 'old':
                    valid_game_path = opt.game_dir + "/" + (
                    str(opt.difficulty_level) + "/" + split  if opt.difficulty_level != '' else opt.game_dir + "/" + split )
                    
                    if opt.graph_emb_type and 'world' in opt.graph_type:
                        valid_manual_world_graphs = load_manual_graphs(valid_game_path + '/' + opt.manual_graph, verbose=False)
                elif opt.dataset_type == 'new':
                    valid_game_path = os.path.join(opt.game_dir, opt.difficulty_level, 'valid', opt.valid)
                else:
                    raise ValueError('This mode is not implemented!')
                    
                if opt.valid_games_num == 5:
                    tmp_valid_game_path = valid_game_path
                else:
                    valid_game_path_list = list(Path(valid_game_path).glob('**/*.ulx'))
                    tmp_valid_game_path = Path(valid_game_path).joinpath('tmp', opt.exp_name + '_' + opt.elements)
                    if not tmp_valid_game_path.exists():
                        tmp_valid_game_path.mkdir(parents=True)
                        valid_game_path_sample = random.sample(valid_game_path_list, opt.valid_games_num)
                        for valid_game_path_ep in valid_game_path_sample:
                            shutil.copy(valid_game_path_ep, tmp_valid_game_path.joinpath(valid_game_path_ep.name))
                            shutil.copy(valid_game_path_ep.with_suffix('.ni'), tmp_valid_game_path.joinpath(valid_game_path_ep.with_suffix('.ni').name))
                            shutil.copy(valid_game_path_ep.with_suffix('.json'), tmp_valid_game_path.joinpath(valid_game_path_ep.with_suffix('.json').name))
                
                tmp_valid_game_path = './' + str(tmp_valid_game_path) + "/"+ opt.game_name
                valid_env, valid_game_file_names = dataset.get_game_env(tmp_valid_game_path, infos_to_request, opt.max_step_per_episode, opt.batch_size, split, verbose=False)
                valid_total_games_count = len(valid_game_file_names)
                # Get Goals as graphs
                valid_goal_graphs = {}
                for game_file in valid_env.gamefiles:
                    valid_goal_graph = get_goal_graph(game_file)
                    if valid_goal_graph:
                        valid_game_id = game_file.split('-')[-1].split('.')[0]
                        valid_goal_graphs[valid_game_id] = valid_goal_graph
                agent.test(opt.batch_size)
                game_names, avg_eps_moves, avg_eps_scores, avg_eps_norm_scores, game_max_scores, reset_eps_count, step_eps_count = episode(no_episode, opt, valid_env, agent, valid_total_games_count, valid_goal_graphs, valid_manual_world_graphs, logger, random_action=random_action)
                writer.add_scalar(f'{split}/norm_scores', np.mean(avg_eps_norm_scores), no_episode)
                writer.add_scalar(f'{split}/steps', np.mean(avg_eps_moves), no_episode)
                # for g, game in enumerate(game_names):
                #     logger.debug(f"Game_#{game} = Score: {avg_eps_scores[g]} Norm_Score: {avg_eps_norm_scores[g]} Moves: {avg_eps_moves[g]}")
                logger.debug(f'{no_episode} episode validation, norm score: {np.mean(avg_eps_norm_scores)}, steps: {np.mean(avg_eps_moves)}')
                if opt.verbose:
                    if split == 'valid' or split == 'in':
                        print("+", end="")
                    elif split == 'test' or split == 'out':
                        print("@", end="")
                    
                if opt.valid_games_num != 5 and Path(tmp_valid_game_path).parent.exists() and 'tmp' in tmp_valid_game_path:
                    shutil.rmtree(Path(tmp_valid_game_path).parent)
                
                # Save Models
                if split == 'valid' or opt.dataset_type == 'new':
                    validation_score, validation_step = np.mean(avg_eps_norm_scores), np.mean(avg_eps_moves)
                    validation_scores.append(validation_score)
                    model_parameters[no_episode] = agent.model.state_dict()
                    optimizer_parameters[no_episode] = agent.optimizer.state_dict()
                    
                    if len(validation_scores) > opt.valid_score_interval:
                        validation_scores.pop(0)
                        _ = model_parameters.pop(no_episode-opt.valid_score_interval)
                        _ = optimizer_parameters.pop(no_episode-opt.valid_score_interval)

                    if len(validation_scores) >= opt.valid_score_interval:
                        mean_validation_score = np.mean(np.array(validation_scores))
                        save_episode = no_episode - (len(validation_scores) - 1 - np.argmax(validation_scores))
                        
                        # Based on score and step
                        if no_episode not in save_episodes and (validation_score > max_validation_score or (validation_score == max_validation_score and validation_step < min_validation_step)):
                            torch.save({'epoch': no_episode, 'model_state_dict': model_parameters[no_episode], 'optimizer_state_dict': optimizer_parameters[no_episode]}, getUniqueFileHandler(opt.checkpoints_filename + f'/episode_{no_episode}', ext='.pt'))
                            torch.save({'validation_scores': validation_scores, 'save_episodes': save_episodes, 'max_mean_validation_score': max_mean_validation_score, 'last_save_episode': last_save_episode, 'max_validation_score': max_validation_score, 'min_validation_step': min_validation_step, 'model_parameters': model_parameters, 'optimizer_parameters': optimizer_parameters}, getUniqueFileHandler(opt.cache_filename + f'/episode_{no_episode}', ext='.pt'))
                            logger.info(f'Saved Model at episode {no_episode} (score/step), score: {validation_score}, step: {validation_step}')
                            max_validation_score, min_validation_step = validation_score, validation_step
                            save_episodes.append(no_episode)
                        
                        # Based on average of scores
                        if mean_validation_score > max_mean_validation_score and save_episode != last_save_episode:
                            logger.info(f'Saved Model at episode {save_episode} (mean), norm scores: {validation_scores}, mean: {mean_validation_score}')
                            max_mean_validation_score = mean_validation_score
                            last_save_episode = save_episode
                            if save_episode not in save_episodes:
                                torch.save({'epoch': save_episode, 'model_state_dict': model_parameters[save_episode], 'optimizer_state_dict': optimizer_parameters[save_episode]}, getUniqueFileHandler(opt.checkpoints_filename + f'/episode_{save_episode}', ext='.pt'))
                                torch.save({'validation_scores': validation_scores, 'save_episodes': save_episodes, 'max_mean_validation_score': max_mean_validation_score, 'last_save_episode': last_save_episode, 'max_validation_score': max_validation_score, 'min_validation_step': min_validation_step, 'model_parameters': model_parameters, 'optimizer_parameters': optimizer_parameters}, getUniqueFileHandler(opt.cache_filename + f'/episode_{no_episode}', ext='.pt'))
                                save_episodes.append(save_episode)
                        
                        # Based on specific episodes
                        if no_episode not in save_episodes and opt.specific_save_episodes is not None and no_episode in opt.specific_save_episodes:
                            logger.info(f'Saved Model at episode {no_episode} (specify), norm scores: {validation_scores}, mean: {mean_validation_score}')
                            if no_episode not in save_episodes:
                                torch.save({'epoch': no_episode, 'model_state_dict': model_parameters[no_episode], 'optimizer_state_dict': optimizer_parameters[no_episode]}, getUniqueFileHandler(opt.checkpoints_filename + f'/episode_{no_episode}', ext='.pt'))
                                torch.save({'validation_scores': validation_scores, 'save_episodes': save_episodes, 'max_mean_validation_score': max_mean_validation_score, 'last_save_episode': last_save_episode, 'max_validation_score': max_validation_score, 'min_validation_step': min_validation_step, 'model_parameters': model_parameters, 'optimizer_parameters': optimizer_parameters}, getUniqueFileHandler(opt.cache_filename + f'/episode_{no_episode}', ext='.pt'))
                                save_episodes.append(no_episode)
                            
                        # # Based on interval
                        # if no_episode % opt.save_model_interval == 0 or no_episode == 100:
                        #     torch.save({'epoch': no_episode, 'model_state_dict': model_parameters[no_episode], 'optimizer_state_dict': optimizer_parametes[no_episode]}, getUniqueFileHandler(opt.interval_checkpoints_filename + f'/episode_{no_episode}', ext='.pt'))
                        #     torch.save({'validation_scores': validation_scores, 'save_episodes': save_episodes, 'max_mean_validation_score': max_mean_validation_score, 'last_save_episode': last_save_episode, 'max_validation_score': max_validation_score, 'min_validation_step': min_validation_step, 'model_parameters': model_parameters, 'optimizer_parameters': optimizer_parameters}, getUniqueFileHandler(opt.cache_filename + f'/episode_{no_episode}', ext='.pt'))
                        #     logger.info(f'Saved Interval Model at episode {no_episode}, norm scores: {validation_scores}, mean: {mean_validation_score}')
                        
                        # Early Stopping
                        if no_episode - last_save_episode > opt.early_stopping:
                            is_early_stopping = True
                    
                    # if no_episode == 100:
                    #     torch.save({'epoch': no_episode, 'model_state_dict': model_parameters[save_episode], 'optimizer_state_dict': optimizer_parametes[save_episode]}, getUniqueFileHandler(opt.checkpoints_filename + f'/episode_{no_episode}', ext='.pt'))
                    #     logger.info(f'Saved Model at episode {no_episode}, norm scores: {validation_scores}, mean: {mean_validation_score}')
            
            if is_early_stopping:
                break
            
            agent.mode = "train"
            agent.model.train()

    env.close()
    game_identifiers = np.array(game_identifiers)
    avg_moves = np.array(avg_moves)
    avg_scores = np.array(avg_scores)
    avg_norm_scores = np.array(avg_norm_scores)
    max_poss_scores = np.array(max_poss_scores)
    reset_counts = np.array(reset_counts)
    setup_counts = np.array(setup_counts)
    if opt.verbose:
        idx = np.apply_along_axis(np.argsort, axis=1, arr=game_identifiers)
        game_avg_moves = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_moves))), axis=0)
        game_norm_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_norm_scores))), axis=0)
        game_avg_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_scores))), axis=0)
        game_avg_reset = np.mean(reset_counts)
        game_avg_setup = np.mean(setup_counts)

        print(f'\nreset_count_ave: {game_avg_reset}; setup_count_ave: {game_avg_setup}')

        msg = "\nGame Stats:\n-----------\n" + "\n".join(
            "  Game_#{} = Score: {:5.2f} Norm_Score: {:5.2f} Moves: {:5.2f}/{}".format(game_no,avg_score,
                                                                                            norm_score, avg_move,
                                                                                            opt.max_step_per_episode)
            for game_no, (norm_score, avg_score, avg_move) in
            enumerate(zip(game_norm_scores, game_avg_scores, game_avg_moves)))

        logger.info(msg)

        total_avg_moves = np.mean(game_avg_moves)
        total_avg_scores = np.mean(game_avg_scores)
        total_norm_scores = np.mean(game_norm_scores)
        msg = opt.mode+" stats: avg. score: {:4.2f}; norm. avg. score: {:4.2f}; avg. steps: {:5.2f}; \n"
        logger.info(msg.format(total_avg_scores, total_norm_scores, total_avg_moves))

        ## Dump log files ......
        str_result = {opt.mode + 'game_ids': game_identifiers, opt.mode + 'max_scores': max_poss_scores,
                      opt.mode + 'scores_runs': avg_scores, opt.mode + 'norm_score_runs': avg_norm_scores,
                      opt.mode + 'moves_runs': avg_moves}

        results_ofile = getUniqueFileHandler(opt.results_filename + '_' +opt.mode+'_results')
        pickle.dump(str_result, results_ofile)
    return avg_scores, avg_norm_scores, avg_moves


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
        
    base_dir = f"{opt.logs_dir}/train/{opt.exp_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    writer_dir = f"{base_dir}/tensorboard/{opt.elements}"
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)
    
    logs_dir = f"{base_dir}/log/"
    os.makedirs(logs_dir, exist_ok=True)
    logger = init_logger(os.path.join(logs_dir, f'{opt.elements}.log'))
    
    if opt.world_evolve_type == 'EbMNbC':
        opt.world_graph_cache = os.path.join(base_dir, 'world_graph_cache')
        os.makedirs(opt.world_graph_cache, exist_ok=True)

    # Reset prune start episodes if pruning is not selected
    if not opt.prune_nodes:
        opt.prune_start_episode = opt.no_train_episodes
    
    opt.amr_save_cache = os.path.join(opt.amr_save_cache, 'train', opt.exp_name + '_' + opt.elements)
    opt.amr_load_cache = os.path.join(opt.amr_load_cache, 'train_base/amr_cache.pkl')
    if os.path.exists(opt.amr_load_cache):
        os.makedirs(opt.amr_save_cache, exist_ok=True)
        shutil.copyfile(opt.amr_load_cache, os.path.join(opt.amr_save_cache, 'base_amr_cache.pkl'))
    
    if opt.world_evolve_type in ['EbM', 'EbMNbC']:
        similar_dict_cache = os.path.join(opt.amr_save_cache, os.path.basename(opt.similar_dict_path))
        shutil.copyfile(opt.similar_dict_path, similar_dict_cache)
        opt.similar_dict_path = similar_dict_cache

    tk_extractor = extractor.get_extractor(opt.token_extractor)

    results_filename = f"{base_dir}/{opt.results_dir}/"
    opt.results_filename = results_filename
    os.makedirs(results_filename, exist_ok=True)
    checkpoints_filename = f"{base_dir}/checkpoints/{opt.elements}"
    opt.checkpoints_filename = checkpoints_filename
    os.makedirs(checkpoints_filename, exist_ok=True)
    interval_checkpoints_filename = f"{base_dir}/interval/{opt.elements}"
    opt.interval_checkpoints_filename = interval_checkpoints_filename
    os.makedirs(interval_checkpoints_filename, exist_ok=True)
    cache_filename = f"{base_dir}/cache/{opt.elements}"
    opt.cache_filename = cache_filename
    os.makedirs(cache_filename, exist_ok=True)
    
    graph = None
    seeds = [random.randint(1, 100) for _ in range(opt.nruns)]
    
    logger.info(opt.exp_name)
    logger.info(opt)
    
    for n in range(0, opt.nruns):

        tokenizer = Tokenizer(noun_only_tokens=opt.noun_only_tokens, use_stopword=opt.use_stopword, ngram=opt.ngram,
                              extractor=tk_extractor)
        rel_extractor = RelationExtractor(tokenizer, openie_url=opt.corenlp_url, amr_server_ip=opt.amr_server_ip, amr_server_port=opt.amr_server_port, amr_save_cache=opt.amr_save_cache, amr_load_cache=opt.amr_load_cache)
        if opt.dropout_ratio == 0.0:
            opt.dropout_ratio = None
        myagent = agent.KnowledgeAwareAgent(graph, opt, tokenizer, rel_extractor, device, writer)
        myagent.type = opt.agent_type
        
        print("Training ...")
        myagent.train(opt.batch_size)  # Tell the agent it should update its parameters.
        opt.mode = "train"
        opt.nepisodes = opt.no_train_episodes  # for training
        opt.max_step_per_episode=opt.train_max_step_per_episode
        starttime = time()
        logger.info('=' * 200)
        logger.info('-' * 90 + f' RUN: {n}, SEED: {opt.seed} ' + '-' * 90)
        logger.info('=' * 200)
        scores, norm_scores, moves = play(myagent, opt, logger, random_action=random_action, writer=writer)
        process_time = time() - starttime
        logger.info("Trained in {} mins {:.2f} secs".format(process_time // 60, process_time % 60))

        # Save train model
        torch.save({'epoch': 'last', 'model_state_dict': myagent.model.state_dict(), 'optimizer_state_dict': myagent.optimizer.state_dict()}, getUniqueFileHandler(checkpoints_filename + '/episode_last', ext='.pt'))

    writer.close()
    if not os.path.exists(os.path.join(opt.amr_save_cache, 'amr_cache.pkl')):
        shutil.rmtree(opt.amr_save_cache)

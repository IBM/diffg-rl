import os
import pickle
import re
from collections import defaultdict
from logging import getLogger
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from textworld import EnvInfos
from torch import optim

import models.scorer_dn as scorer_dn
from utils.extractor import any_substring_extraction
from utils.generic import load_embeddings, max_len, to_tensor
from utils.kg import (add_triplets_to_graph, construct_graph,
                      construct_graph_amr, khop_neighbor_graph,
                      shortest_path_subgraph, similar_entity_context_subgraph,
                      similar_entity_subgraph)
from utils.textworld_utils import (process_full_facts, process_step_facts,
                                   serialize_facts)

# Agent must have train(), test(), act() functions and infos_to_request as properties
logger = getLogger("log").getChild("sub")

class KnowledgeAwareAgent:
    """ Knowledgeable Neural Agent for playing TextWorld games. """
    # UPDATE_FREQUENCY = 20
    LOG_FREQUENCY = 20 # in episodes
    GAMMA = 0.9
    type = "KnowledgeAware"

    def __init__(self, graph, opt, tokenizer=None, rel_extractor = None, device=None, writer=None) -> None:
        print("Initializing Knowledge-Aware Neural Agent")
        self.seed = opt.seed
        self.hidden_size=opt.hidden_size
        self.device = device
        self.local_evolve_type = opt.local_evolve_type
        self.world_evolve_type = opt.world_evolve_type
        self._initialized = False
        self._epsiode_has_started = False
        self.sentinel_node = True # Sentinel node is added to local/world to allow attention module
        self.epsilon = opt.egreedy_epsilon
        self.tokenizer = tokenizer
        self.rel_extractor = rel_extractor
        self.pruned_concepts = []

        self.emb_loc = opt.emb_loc
        self.word_emb_type = opt.word_emb_type
        self.graph_emb_type = opt.graph_emb_type
        self.word_emb, self.graph_emb = None, None
        self.word2id, self.node2id = {}, {}
        self.similar_alias = opt.similar_alias
        self.dropout_ratio = opt.dropout_ratio
        self.prune_rate = opt.prune_rate
        self.layer_norm = opt.layer_norm
        self.diff_network = opt.diff_network
        self.value_network = opt.value_network
        self.curiosity = opt.curiosity
        self.reward_attenuation = opt.reward_attenuation
        self.update_frequency = opt.reward_update_frequency
        self.observation_list = []
        self.skip_cmd = opt.skip_cmd
        self.nhead = opt.nhead
        self.difficulty = opt.difficulty_level
        self.use_edge = False
        self.diff_no_elu = opt.diff_no_elu

        self._episode_has_started = False

        if self.word_emb_type is not None:
            self.word_emb = load_embeddings(self.emb_loc, self.word_emb_type)
            self.word_vocab = self.word_emb.key_to_index
            for i, w in enumerate(self.word_vocab):
                self.word2id[w] = i
        # Graphs
        self.graph_type = opt.graph_type
        self.reset_graph()

        if self.graph_emb_type is not None and ('local' in self.graph_type or 'world' in self.graph_type):
            self.graph_emb = load_embeddings(self.emb_loc, self.graph_emb_type)
            self.kg_graph = graph
            self.node_vocab = self.graph_emb.key_to_index
            for i, w in enumerate(self.node_vocab):
                self.node2id[w] = i
        
        self.model = scorer_dn.CommandScorerWithDN(self.word_emb, 
                                            hidden_size=self.hidden_size, device=device, dropout_ratio=self.dropout_ratio, word2id=self.word2id, value_network=self.value_network, diff_network=self.diff_network,
                                            diff_no_elu=self.diff_no_elu)
            
        if torch.cuda.is_available():
            print('Convert model to GPU')
            self.model.to(device)
        if opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), 0.00003)
        elif opt.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters())
        self.hist_scmds_size = opt.hist_scmds_size
        self.stats = {"episode": defaultdict(list), "game": defaultdict(list)}
        self.mode = "test"
        self.similar_dict = None
        
        self.exp_name = opt.exp_name
        if self.world_evolve_type == 'EbMNbC':
            self.world_graph_cache = opt.world_graph_cache
        
        self.wrong_texts = defaultdict(int)
        self.wrong_outs = {}
        self.train_iteration = 0
        self.writer = writer

    def start_episode(self, batch_size):
        # Called at the beginning of each episode
        self._episode_has_started = True
        if self.mode == 'train':
            self.no_train_episodes += 1
            self.transitions = [[] for _ in range(batch_size)]
            self.stats["game"] = defaultdict(list)
        self.reset_parameters(batch_size)
        if self.curiosity > 0:
            self.observation_list = []

    def end_episode(self):
        # Called at the end of each episode
        self._episode_has_started = False
        self.reset_graph()
        if self.curiosity > 0:
            self.observation_list = []

        if self.mode == 'train':
            for k, v in self.stats["game"].items():
                self.stats["episode"][k].append(np.mean(v, axis=0))
            if self.no_train_episodes % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_episodes)
                msg += "  " + "  ".join("{}: {:5.2f}".format(k, np.mean(v,axis=0)) for k, v in self.stats["episode"].items())
                print(msg)
                self.stats["episode"] = defaultdict(list) # reset stat

    def train(self, batch_size=1):
        self.mode = "train"
        self.model.train()
        self.no_train_step = 0
        self.no_train_episodes = 0

    def test(self,batch_size=1):
        self.mode = "test"
        self.model.eval()
        if self.value_network == 'obs' or self.diff_network == 'none':
            self.model.reset_hidden(batch_size)

    def reset_parameters(self, batch_size):
        # Called at the beginning of each batch
        self.agent_loc = ['' for _ in range(batch_size)]
        self.last_done = [False] * batch_size
        if self.mode == 'train':
            self.last_score = tuple([0.0] * batch_size)
            self.batch_stats = [{"max": defaultdict(list), "mean": defaultdict(list)} for _ in range(batch_size)]
        if self.value_network == 'obs' or self.diff_network == 'none':
            self.model.reset_hidden(batch_size)
        self.reset_graph()

    def reset_graph(self):
        self.world_graph = {}
        self.local_graph = {}
        self.rel_extractor.agent_loc = ''
        if self.local_evolve_type == 'amr':
            self.current_facts = defaultdict(dict)
        else:
            self.current_facts = defaultdict(set)  # unserialized facts, use serialize_facts() from utils
        self.object_entities = defaultdict(list)
        self.container_entities = defaultdict(list)
        self.appeared_commands = defaultdict(list)
        self.room = defaultdict(str)
        self.goals = {}
        self.entities_in_obs = []

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,won=True, lost=True,location = True,
                        last_action=True,game=True,facts=True,entities=True) # Last line needed for ground truth local graph

    def get_local_graph(self, batch, obs, hint, infos, cmd, prev_facts, graph_mode, object_entities, container_entities, inventory, step_no=0):
        processed_text_list = []
        if graph_mode == "full":
            current_facts = process_full_facts(infos["game"], infos["facts"])
            current_triplets = serialize_facts(current_facts)  # planning graph triplets
            local_graph, entities = construct_graph(current_triplets)
        else:
            if self.local_evolve_type == 'direct': # Similar to KG-DQN
                state = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
                hint_str = " ".join(hint)
                prev_action = cmd
                if cmd == 'restart':
                    prev_action = None
                local_graph, current_facts = self.rel_extractor.fetch_triplets(state+hint_str, infos["local_graph"], prev_action=prev_action)
                entities = self.rel_extractor.kg_vocab
            elif self.local_evolve_type == 'sgp':
                state = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
                hint_str = " ".join(hint)
                prev_action = cmd
                if cmd == 'restart':
                    prev_action = None
                local_graph, current_facts = self.rel_extractor.fetch_triplets_sgp(state+hint_str, infos["local_graph"], prev_action=prev_action)
                entities = self.rel_extractor.kg_vocab
            elif self.local_evolve_type == 'amr':
                if len(inventory.split('\n')) > 1:
                    sub_verb, objs = inventory.split(':\n')[0], inventory.split(':\n')[1].split('\n')
                    inventory = ''
                    for obj in objs:
                        inventory += sub_verb + ' ' + obj.strip() + '.' + ' '
                else:
                    pass
                if step_no == 0:
                    state = infos['description'] + '. ' + inventory
                else:
                    state = obs + '. ' + inventory
                    
                prev_facts_for_amr = {entity: fact[0] for entity, fact in prev_facts.items()}
                current_facts_from_amr, room, processed_text_list = self.rel_extractor.fetch_triplets_amr(state, prev_facts_for_amr, object_entities, container_entities, self.room[batch]) # fetch_triplets_amr is in kg.py
                current_facts = {}
                for entity in current_facts_from_amr:
                    if entity in prev_facts.keys() and prev_facts_for_amr[entity][1] == current_facts_from_amr[entity][1]:
                        current_facts[entity] = (current_facts_from_amr[entity], prev_facts[entity][1])
                    else:
                        current_facts[entity] = (current_facts_from_amr[entity], step_no)
                self.room[batch] = room
                local_graph, entities = construct_graph_amr(current_facts.values())
            else: # Ground-Truth, from textworld
                current_facts = process_step_facts(prev_facts, infos["game"], infos["facts"],
                                                    infos["last_action"], cmd)
                current_triplets = serialize_facts(current_facts)  # planning graph triplets
                local_graph, entities = construct_graph(current_triplets)
        return local_graph, current_facts, entities, processed_text_list

    def get_world_graph(self, obs, hint, infos, graph_mode, prune_nodes, object_entities, container_entities, step_no=0):
        # hints could be list of goals or recipe to follow.
        # Options to choose for evolve graph: DC, CDC,  neighbours (with/ without pruning), manual
        add_edges = []
        if graph_mode == "full":
            if 'goal_graph' in infos and infos['goal_graph']:
                add_edges = [[e.replace('_', ' ') for e in edge]+["AtLocation"] for edge in infos['goal_graph'].edges]

            entities = []
            prev_entities = infos["entities"]
            for entity in prev_entities:
                et_arr = re.split(r'[- ]+', entity)
                entity_nodes = any_substring_extraction(entity, self.kg_graph, ngram=len(et_arr))
                jentity = '_'.join(et_arr)

                if not entity_nodes: # No valid entry in the kg_graph
                    entity_nodes = set(jentity)
                for en in entity_nodes:
                    if en != jentity:
                        add_edges.append([en, jentity, "RelatedTo"])
                entities.extend(entity_nodes)
            graph = shortest_path_subgraph(self.kg_graph, nx.DiGraph(), entities)
            world_graph, entities = add_triplets_to_graph(graph, add_edges)
        else:
            prev_entities = list(infos["world_graph"].nodes) if infos["world_graph"] else []
            world_graph = infos["world_graph"]
            
            # Extract new entities (Entities that have not appeared in previous steps)
            if self.world_evolve_type == 'EbM':
                base_entities = self.tokenizer.extract_world_graph_base_entities(infos["entities"], self.kg_graph)
                new_entities = base_entities - set(prev_entities + self.tokenizer.ignore_list + self.pruned_concepts)
                new_entities = list(set(world_graph.nodes) ^ new_entities)
            elif self.world_evolve_type == 'EbMNbC':
                new_entities = list(set(world_graph.nodes) ^ set(object_entities + container_entities))
            else:
                state = "{}\n{}".format(obs, infos["description"])
                hint_str = " ".join(hint)
                state_entities = self.tokenizer.extract_world_graph_entities(state, self.kg_graph)
                hint_entities = self.tokenizer.extract_world_graph_entities(hint_str, self.kg_graph)
                inventory_entities = self.tokenizer.extract_world_graph_entities(infos["inventory"], self.kg_graph)
                new_entities = list((state_entities | hint_entities | inventory_entities) - set(prev_entities + self.tokenizer.ignore_list + self.pruned_concepts))

            node_weights = {}
            if not nx.is_empty(world_graph):
                node_weights = nx.get_node_attributes(world_graph, 'weight')
                # if 'sentinel_weight' in world_graph.graph:
                    # sentinel_weight = world_graph.graph['sentinel_weight']
            if self.world_evolve_type == 'DC':
                entities = prev_entities + new_entities
                world_graph = self.kg_graph.subgraph(entities).copy()
            elif 'NG' in self.world_evolve_type: # Expensive option
                if new_entities:
                    # Setting max_khop_degree to higher value results in adding high-degree nodes ==> noise
                    # cutoff =1 select paths of length 2 between every pair of nodes.
                    new_graph = khop_neighbor_graph(self.kg_graph, new_entities, cutoff=1,max_khop_degree=100)
                    world_graph = nx.compose(world_graph, new_graph)
            elif self.world_evolve_type == 'manual':
                assert ('manual_world_graph' in infos and infos['manual_world_graph'] and 'graph' in infos[
                    'manual_world_graph']), 'No valid manual world graph found. Use other options'
                select_entities = list(set(infos['manual_world_graph']['entities']).intersection(set(new_entities)))
                new_graph = khop_neighbor_graph(infos['manual_world_graph']['graph'], select_entities, cutoff=1)
                world_graph = nx.compose(world_graph, new_graph)
            elif self.world_evolve_type == 'CDC': # default options = CDC
                inventory_entities = self.tokenizer.extract_world_graph_entities(infos["inventory"], self.kg_graph)
                if new_entities or inventory_entities:
                    command_entities=[]
                    for cmd in infos['admissible_commands']:
                        if 'put' in cmd or 'insert' in cmd:
                            entities = self.tokenizer.extract_world_graph_entities(cmd, self.kg_graph)
                            command_entities.extend(entities)
                    world_graph = shortest_path_subgraph(self.kg_graph, world_graph, new_entities,
                                                         inventory_entities, command_entities)
            elif self.world_evolve_type == 'EbM': # Extracting by Meaning
                if new_entities:
                    world_graph = similar_entity_subgraph(self.kg_graph, world_graph, object_entities+container_entities, self.similar_dict, alias=self.similar_alias)
            elif self.world_evolve_type == 'EbMNbC':# Extracting by Meaning + Narrowing by Circumstances
                game_id = infos['game'].metadata["uuid"].split("-")[-1]
                world_graph_cache_path = os.path.join(self.world_graph_cache, game_id+'pkl')
                if new_entities:
                    if self.difficulty == 'hard' or not os.path.exists(world_graph_cache_path):
                        world_graph = similar_entity_context_subgraph(self.kg_graph, world_graph, object_entities, container_entities, self.similar_dict, alias=self.similar_alias, prune_rate=self.prune_rate, step=step_no)
                        if self.difficulty != 'hard':
                            # Use cache to speed up the process except for HARD
                            game_id = infos['game'].metadata["uuid"].split("-")[-1]
                            world_graph_cache_path = os.path.join(self.world_graph_cache, game_id + '.pkl')
                            with open(world_graph_cache_path, 'wb') as f:
                                pickle.dump(world_graph, f)
                    else:
                        with open(world_graph_cache_path, 'rb') as f:
                            world_graph = pickle.load(f)
            elif self.world_evolve_type == 'goal':
                world_graph = infos['goal_graph']
                for (s, o) in infos['goal_graph'].edges:
                    world_graph.edges[s, o]['relation'] = 'atlocation'
            else:
                raise ValueError('This method has not been implemented.')

            # Prune Nodes
            if prune_nodes and not nx.is_empty(world_graph) and len(
                    world_graph.nodes) > 10:
                prune_count = int(len(world_graph.nodes) / 30)
                for _ in range(prune_count):
                    if any(node_weights):
                        rnode = min(node_weights, key=node_weights.get)
                        self.pruned_concepts.append(rnode)
                        # print('pruning ' + rnode)
                        world_graph.graph['sentinel_weight'] += node_weights[rnode]
                        if rnode in world_graph:
                            world_graph.remove_node(rnode)

            world_graph.remove_edges_from(nx.selfloop_edges(world_graph))
            entities = list(world_graph.nodes)
        return world_graph, entities

    def update_current_graph(self, obs, cmd, hints, infos, graph_mode, prune_nodes=False, step_no=0, goal_world=False):
        # hints could be list of goals or recipe to follow.
        batch_size = len(obs)
        info_per_batch = [{k: v[i] for k, v in infos.items()} for i in range(len(obs))]
        for b in range(batch_size):
            
            if step_no == 0:
                self.goals[b] = infos['goal_graph'][b]
            
            if self.skip_cmd and (cmd[b] == 'look' or 'examine' in cmd[b] or 'open' in cmd[b] or 'close' in cmd[b]):
                continue

            new_commands = set(infos['admissible_commands'][b]) - set(self.appeared_commands[b])
            if ('local' in self.graph_type or self.world_evolve_type in ['EbM', 'EbMNbC']) and new_commands:
                object_entities, container_entities = self.rel_extractor.extract_entities_from_commands(infos['admissible_commands'][b])
                self.object_entities[b] = list(set(object_entities) | set(self.object_entities[b]))
                self.container_entities[b] = list((set(container_entities) | set(self.container_entities[b])) - set(self.object_entities[b]))
                self.appeared_commands[b].extend(new_commands)
                # self.check_entities(b, self.object_entities[b] + self.container_entities[b], list(set(infos["entities"][b]) - set(['east', 'west', 'north', 'south'])), obs[b] + ' ' + infos['inventory'][b], infos)

            if 'local' in self.graph_type:
                self.rel_extractor.agent_loc=self.agent_loc[b]
                info_per_batch[b]["local_graph"] = self.local_graph[b] if b in self.local_graph else nx.DiGraph()
                local_graph, current_facts, _, processed_text_list = self.get_local_graph(b, obs[b], hints[b], info_per_batch[b], cmd[b], self.current_facts[b], graph_mode, self.object_entities[b], self.container_entities[b], infos['inventory'][b], step_no=step_no)
                self.agent_loc[b] = self.rel_extractor.agent_loc
                self.local_graph[b] = local_graph
                self.current_facts[b] = current_facts
                self.check_local_graph(b, infos["facts"][b], self.local_graph[b], processed_text_list, infos, obs[b]) # (answer, generated)

            if 'world' in self.graph_type:
                info_per_batch[b]["world_graph"] = self.world_graph[b] if b in self.world_graph else nx.DiGraph()
                # info_per_batch[b]["goal_graph"] = infos["goal_graph"][b] if 'goal_graph' in infos else None
                world_graph, _ = self.get_world_graph(obs[b], hints[b], info_per_batch[b], graph_mode, prune_nodes, self.object_entities[b], self.container_entities[b], step_no=step_no)

                self.world_graph[b] = world_graph
            
            if goal_world:
                gaol_world_graph = nx.Graph()
                for (s_g, o_g) in list(infos["goal_graph"][b].edges):
                    for (s_w, o_w) in list(world_graph.edges):
                        if s_g == s_w and o_g == o_w:
                            gaol_world_graph.add_edge(s_w, o_w)
                            break
                self.world_graph[b] = gaol_world_graph

    #### Check the generated things ####

    def check_local_graph(self, b, answers, candidate, processed_text_list, infos, obs):
        # Check if the local graph generated by AMR is correct
        out = {}
        for ans in answers:
            if ans.name in ['on', 'in']:
                subject, object = ans.names
                subject, object = subject.lower(), object.lower()
                if object == 'i':
                    object = 'you'
                if subject in self.entities_in_obs:
                    if subject in list(candidate.nodes):
                        candidate_object = list(candidate.succ[subject])[0]
                        if object != candidate_object:
                            out[subject] = (object, candidate_object)
                    else:
                        out[subject] = (object, '')
        if len(out) > 0:
            processed = {}
            save_cache = False
            for entity, locations in out.items():
                for processed_text in processed_text_list:
                    if entity in processed_text and entity in processed_text and processed_text in self.rel_extractor.amr_rest.cache.keys() and self.wrong_texts[processed_text] < 10:
                        self.rel_extractor.amr_rest.delete_cache(processed_text)
                        self.wrong_texts[processed_text] += 1
                        processed[processed_text] = self.wrong_texts[processed_text]
                        save_cache = True
            if save_cache:
                self.rel_extractor.amr_rest.save_cache()
                with open('./wrong_cache.pkl', 'rb') as f:
                    wrong_cache = pickle.load(f)
                wrong_cache.update(self.rel_extractor.amr_rest.wrong_cache)
                with open('./wrong_cache.pkl', 'wb') as f:
                    pickle.dump(wrong_cache, f)
            
            # if len(out) > len(self.wrong_outs):
                logger.debug('-' * 30 + ' Extraction Local Graph Error ' + '-' * 30)
                logger.debug('wrong dict')
                logger.debug(out)
                logger.debug('processed_text')
                logger.debug(processed)
                logger.debug('correct facts')
                logger.debug(infos["facts"][b])
                logger.debug('generated graphs:')
                logger.debug(dict(self.local_graph[b].edges))
                logger.debug('generated object entities:')
                logger.debug(self.object_entities[b])
                logger.debug('generated container entities')
                logger.debug(self.container_entities[b])
                logger.debug('observation')
                logger.debug(obs)
                logger.debug('inventory:')
                logger.debug(infos['inventory'][b])
                logger.debug('correct entities:')
                logger.debug(infos["entities"][b])
                logger.debug('admissible command:')
                logger.debug(infos['admissible_commands'][b])
                # raise RuntimeError('extraction local graph is failed.')
            
            self.wrong_outs = out
    
    def check_entities(self, b, generated_entities, correct_entities, obs, infos):
        # # Check if the entities generated by AMR is correct
        entities_in_obs = []
        for entity in correct_entities:
            if (' ' + entity.lower()) in obs.lower() and 'door' not in entity.lower():
                entities_in_obs.append(entity.lower())
        self.entities_in_obs.extend(entities_in_obs)
        wrong_entities_generated = set(generated_entities) - set(self.entities_in_obs)
        wrong_entities_not_generated = set(self.entities_in_obs) - set(generated_entities)
        if len(wrong_entities_generated) != 0 or len(wrong_entities_not_generated) != 0:
            logger.debug('-' * 30 + ' Extraction Entities Error ' + '-' * 30)
            logger.debug('wrong entities generated:')
            logger.debug(wrong_entities_generated)
            logger.debug('wrong entities not generated:' )
            logger.debug(wrong_entities_not_generated)
            logger.debug('correct entities:')
            logger.debug(infos["entities"][b])
            logger.debug('generated object entities:')
            logger.debug(self.object_entities[b])
            logger.debug('generated container entities')
            logger.debug(self.container_entities[b])
            logger.debug('admissible command:')
            logger.debug(infos['admissible_commands'][b])
            logger.debug('observation')
            logger.debug(obs)
            logger.debug('inventory:')
            logger.debug(infos['inventory'][b])
            # raise RuntimeError('extraction entities is failed.')

    #### Process Series ####
    
    def _process(self, texts_, vocabulary, sentinel=False, truncate=False, world=False):
        # texts = list(map(self.extract_entity_ids, texts))
        if world:
            if len(texts_) == 0:
                return to_tensor(np.ones((1, 1)) * vocabulary["<PAD>"], self.device)
            texts = [self.tokenizer.extract_world_entity_ids(word, vocabulary) for word in texts_]
        else:
            texts = [self.tokenizer.extract_entity_ids(word, vocabulary, truncate) for word in texts_]
        max_len = max(len(l) for l in texts)
        if self.use_edge and max_len > 1:
            max_len = 1
            texts = [[t[0]] for t in texts]
        num_items = len(texts) + 1 if sentinel else len(texts)  # Add sentinel entry here for the attention mechanism
        if "<PAD>" in vocabulary:
            padded = np.ones((num_items, max_len)) * vocabulary["<PAD>"]
        else:
            print('Warning: No <PAD> found in the embedding vocabulary. Using the id:0 for now.')
            padded = np.zeros((num_items, max_len))

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = to_tensor(padded, self.device)
        return padded_tensor
    
    def _process_diff_entity(self, texts_, vocabulary, sentinel=False):
        if len(texts_) == 0:
            padded = np.ones((1, 1)) * vocabulary["<PAD>"]
            padded_tensor = to_tensor(padded, self.device)
            return padded_tensor
        texts = [self.tokenizer.extract_diff_entity_ids(word, vocabulary) for word in texts_]
        try:
            max_len = max(len(l) for l in texts)
        except:
            raise Exception(f'diff is empty, object: {self.object_entities}, container: {self.container_entities}, local: {self.local_graph[0].edges}, global: {self.world_graph[0].edges}')
        num_items = len(texts) + 1 if sentinel else len(texts)  # Add sentinel entry here for the attention mechanism
        if "<PAD>" in vocabulary:
            padded = np.ones((num_items, max_len)) * vocabulary["<PAD>"]
        else:
            print('Warning: No <PAD> found in the embedding vocabulary. Using the id:0 for now.')
            padded = np.zeros((num_items, max_len))

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = to_tensor(padded, self.device)
        return padded_tensor
    
    def _process_dn(self, diffs, vocabulary, obs=None, sentinel=False):
        entities_ = []
        locals_ = []
        worlds_ = []
        local_locations_ = []
        world_locations_ = []
        for diff in diffs:
            entities_.append(diff['entity'])
            locals_.append(diff['current'][0])
            if len(diff['current']) > 1:
                local_locations_.append(diff['current'][1])
            worlds_.append([candidate[0] for candidate in diff['candidates']])
            world_locations_.append([candidate[1] for candidate in diff['candidates']])
    
        entities = self._process_diff_entity(entities_, vocabulary, sentinel=sentinel)
        locals = self._process_diff_entity(locals_, vocabulary, sentinel=sentinel)
        local_locations = self._process_diff_entity(local_locations_, vocabulary, sentinel=sentinel)
            
        worlds = []
        world_locations = []
        for i, world_ in enumerate(worlds_):
            worlds.append(self._process_diff_entity(world_, vocabulary, sentinel=sentinel))
            world_locations.append(self._process_diff_entity(world_locations_[i], vocabulary, sentinel=sentinel))
        
        if len(worlds) == 0:
            worlds = np.ones((1, 1)) * vocabulary["<PAD>"]
            worlds = [to_tensor(worlds, self.device)]
        
        if len(world_locations) == 0:
            world_locations = np.ones((1, 1)) * vocabulary["<PAD>"]
            world_locations = [to_tensor(world_locations, self.device)]
        
        return entities, locals, worlds

    def _discount_rewards(self, batch_id, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions[batch_id]))):
            rewards, _, _, values, _ = self.transitions[batch_id][t]
            R = torch.tensor(rewards) + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any], scored_commands: list, random_action=False, truncate=False, no_episode=0):
        batch_size = len(obs)
        if not self._episode_has_started:
            self.start_episode(batch_size)

        just_finished = [done[b] != self.last_done[b] for b in range(batch_size)]
        sel_rand_action_idx = [np.random.choice(len(infos["admissible_commands"][b])) for b in range(batch_size)]
        if random_action:
            return [infos["admissible_commands"][b][sel_rand_action_idx[b]] for b in range(batch_size)]

        torch.autograd.set_detect_anomaly(True)
        input_t = []
        # Build agent's observation: feedback + look + inventory.
        state = ["{}\n{}\n{}\n{}".format(obs[b], infos["description"][b], infos["inventory"][b], ' \n'.join(
                        scored_commands[b])) for b in range(batch_size)]
        # Tokenize and pad the input and the commands to chose from.
        state_tensor = self._process(state, self.word2id, truncate=truncate)

        command_list = []
        for b in range(batch_size):
            cmd_b = self._process(infos["admissible_commands"][b], self.word2id, truncate=truncate)
            command_list.append(cmd_b)
        max_num_candidate = max_len(infos["admissible_commands"])
        max_num_word = max([cmd.size(1) for cmd in command_list])
        commands_tensor = to_tensor(np.zeros((batch_size, max_num_candidate, max_num_word)), self.device)
        for b in range(batch_size):
            commands_tensor[b,:command_list[b].size(0), :command_list[b].size(1)] = command_list[b]

        localkg_tensor = torch.FloatTensor()
        localkg_adj_tensor = torch.FloatTensor()
        worldkg_tensor = torch.FloatTensor()
        worldkg_adj_tensor = torch.FloatTensor()
        localkg_hint_tensor = torch.FloatTensor()
        worldkg_hint_tensor = torch.FloatTensor()
        world_pre_tensor = torch.FloatTensor()
        if self.graph_emb_type is not None and ('local' in self.graph_type or 'world' in self.graph_type):

            # prepare Local graph and world graph ....
            # Extra empty node (sentinel node) for no attention option
            #  (Xiong et al ICLR 2017 and https://arxiv.org/pdf/1612.01887.pdf)
            if 'world' in self.graph_type:
                world_entities = []
                for b in range(batch_size):
                    world_entities.extend(self.world_graph[b].nodes())
                world_entities = set(world_entities)
                wentities2id = dict(zip(world_entities,range(len(world_entities))))
                max_num_nodes = len(wentities2id) + 1 if self.sentinel_node else len(wentities2id)
                worldkg_tensor = self._process(wentities2id, self.node2id, sentinel=self.sentinel_node, truncate=truncate, world=True)
                world_adj_matrix = np.zeros((batch_size, max_num_nodes, max_num_nodes), dtype="float32")
                for b in range(batch_size):
                    # get adjacentry matrix for each batch based on the all_entities
                    triplets = [list(edges) for edges in self.world_graph[b].edges.data('relation')]
                    for [e1, e2, r] in triplets:
                        e1 = wentities2id[e1]
                        e2 = wentities2id[e2]
                        world_adj_matrix[b][e1][e2] = 1.0
                        world_adj_matrix[b][e2][e1] = 1.0 # reverse relation
                    for e1 in list(self.world_graph[b].nodes):
                        e1 = wentities2id[e1]
                        world_adj_matrix[b][e1][e1] = 1.0
                    if self.sentinel_node: # Fully connected sentinel
                        world_adj_matrix[b][-1,:] = np.ones((max_num_nodes),dtype="float32")
                        world_adj_matrix[b][:,-1] = np.ones((max_num_nodes), dtype="float32")
                worldkg_adj_tensor = to_tensor(world_adj_matrix, self.device, type="float")
                if self.use_edge:
                    world_pre_tensor = torch.zeros(batch_size, max_num_nodes, max_num_nodes, 1, dtype=torch.int64, device=self.device)
                    for b in range(batch_size):
                        # get adjacentry matrix for each batch based on the all_entities
                        batch_pre_matrix = [['' for _ in range(max_num_nodes)] for _ in range(max_num_nodes)]
                        triplets = [list(edges) for edges in self.world_graph[b].edges.data('relation')]
                        for [e1, e2, r] in triplets:
                            e1 = wentities2id[e1]
                            e2 = wentities2id[e2]
                            batch_pre_matrix[e1][e2] = r
                            batch_pre_matrix[e2][e1] = r # reverse re
                        for i, pre_list in enumerate(batch_pre_matrix):
                            world_pre_tensor[b][i] = self._process(pre_list, self.node2id, truncate=truncate)

            if 'local' in self.graph_type:
                local_entities = []
                for b in range(batch_size):
                    local_entities.extend(self.local_graph[b].nodes())
                local_entities = set(local_entities)
                lentities2id = dict(zip(local_entities,range(len(local_entities))))
                max_num_nodes = len(lentities2id) + 1 if self.sentinel_node else len(lentities2id)
                localkg_tensor = self._process(lentities2id, self.word2id, sentinel=self.sentinel_node, truncate=truncate)
                local_adj_matrix = np.zeros((batch_size, max_num_nodes, max_num_nodes), dtype="float32")
                for b in range(batch_size):
                    # get adjacentry matrix for each batch based on the all_entities
                    triplets = [list(edges) for edges in self.local_graph[b].edges.data('relation')]
                    for [e1, e2, r] in triplets:
                        e1 = lentities2id[e1]
                        e2 = lentities2id[e2]
                        local_adj_matrix[b][e1][e2] = 1.0
                        local_adj_matrix[b][e2][e1] = 1.0
                    for e1 in list(self.local_graph[b].nodes):
                        e1 = lentities2id[e1]
                        local_adj_matrix[b][e1][e1] = 1.0
                    if self.sentinel_node:
                        local_adj_matrix[b][-1,:] = np.ones((max_num_nodes),dtype="float32")
                        local_adj_matrix[b][:,-1] = np.ones((max_num_nodes), dtype="float32")
                localkg_adj_tensor = to_tensor(local_adj_matrix, self.device, type="float")
            
            if self.diff_network != 'none':
                entities_list = []
                locals_list = []
                worlds_list = []
                for b in range(batch_size):
                    diff_b = []
                    for object_entity in self.object_entities[b]:
                        if len(list(self.local_graph[b].succ[object_entity])) == 0:
                            current = []
                        else:
                            current = list(self.local_graph[b].succ[object_entity])[0]
                            try:
                                current_locations = list(self.local_graph[b].succ[current])
                            except KeyError:
                                current_locations = []
                            if len(current_locations) > 0:
                                current = [current, current_locations[0]]
                            else:
                                current = [current]
                            
                        try:
                            candidates = list(self.world_graph[b].succ[object_entity])
                        except KeyError:
                            candidates = []
                        candidate_trees = []
                        for candidate in candidates:
                            try:
                                candidate_locations = list(self.local_graph[b].succ[candidate])
                            except KeyError:
                                candidate_locations = []
                            if len(candidate_locations) > 0:
                                candidate_trees.append([candidate, list(self.local_graph[b].succ[candidate])[0]])
                            else:
                                candidate_trees.append([candidate])
                                
                        diff = {'entity': object_entity, 'current': current, 'candidates': candidate_trees}
                        diff_b.append(diff)
                                                        
                    entities_b, locals_b, worlds_b = self._process_dn(diff_b, self.word2id, obs=obs[b]+infos['inventory'][b])
                    entities_list.append(entities_b)
                    locals_list.append(locals_b)
                    worlds_list.append(worlds_b)
                                    
                num_object_entities =  max(max_len(list(self.object_entities.values())), 1)
                max_num_entities = max([ent.size(1) for ent in entities_list])
                max_num_locals = max([loc.size(1) for loc in locals_list])
                entities_tensor = to_tensor(np.zeros((batch_size, num_object_entities, max_num_entities)), self.device)
                locals_tensor = to_tensor(np.zeros((batch_size, num_object_entities, max_num_locals)), self.device)
                for b in range(batch_size):
                    entities_tensor[b,:entities_list[b].size(0), :entities_list[b].size(1)] = entities_list[b]
                    locals_tensor[b,:locals_list[b].size(0), :locals_list[b].size(1)] = locals_list[b]
                            

            if len(scored_commands) > 0:
                # Get the scored commands as one string
                hint_str = [' \n'.join(
                        scored_commands[b][-self.hist_scmds_size:]) for b in range(batch_size)]
            else:
                hint_str = [obs[b] + ' \n' + infos["inventory"][b] for b in range(batch_size)]
            localkg_hint_tensor = self._process(hint_str, self.word2id, truncate=truncate)
            worldkg_hint_tensor = self._process(hint_str, self.node2id, truncate=truncate)

        input_t.append(state_tensor)
        input_t.append(commands_tensor)
        input_t.append(localkg_tensor)
        input_t.append(localkg_adj_tensor)
        input_t.append(entities_tensor)
        input_t.append(locals_tensor)
        input_t.append(worlds_list)

        outputs, _indexes, _values = self.model(*input_t)
        indexes, values = _indexes.view(batch_size), _values.view(batch_size)
        sel_action_idx = [indexes[b] for b in range(batch_size)]
        action = [infos["admissible_commands"][b][sel_action_idx[b]] for b in range(batch_size)]

        if any(done):
            for b in range(batch_size):
                if done[b]:
                    if self.value_network == 'obs' or self.diff_network == 'none':
                        self.model.reset_hidden_per_batch(b)
                    action[b] = 'look'

        if self.mode == "test":
            self.last_done = done
            return action

        self.no_train_step += 1
        last_score = list(self.last_score)
        policy_losses, value_losses, entropies, advantages = [], [], [], []
        batch_loss_list = []
        batch_count = 0
        self.train_iteration += 1
        for b, score_b in enumerate(score):
            # Update local/world graph attention weights
            if self.diff_network == 'none' and 'world' in self.graph_type:
                with torch.no_grad():
                    att_wts = self.model.world_attention[b].flatten().cpu().numpy()
                edge_attr = dict(zip(wentities2id.keys(),att_wts))
                nx.set_node_attributes(self.world_graph[b], edge_attr, 'weight')
                self.world_graph[b].graph["sentinel_weight"] = att_wts[-1]
            if self.diff_network == 'none' and 'local' in self.graph_type:
                with torch.no_grad():
                    att_wts = self.model.local_attention[b].flatten().cpu().numpy()
                edge_attr = dict(zip(lentities2id.keys(),att_wts))
                nx.set_node_attributes(self.local_graph[b], edge_attr, 'weight')
                self.local_graph[b].graph["sentinel_weight"] = att_wts[-1]
            if self.transitions[b]:
                reward = (score_b - last_score[b])
                reward = reward + 100 if infos["won"][b] else reward
                reward = reward - 100 if infos["lost"][b] else reward
                if self.curiosity > 0 and obs[b] not in self.observation_list:
                    reward = reward + self.curiosity
                    self.observation_list.append(obs[b])
                if self.reward_attenuation > 0:
                    reward = reward - self.reward_attenuation
                self.transitions[b][-1][0] = reward  # Update reward information.
                last_score[b] = score_b
            if self.no_train_step % self.update_frequency == 0 or just_finished[b]:
                # Update model
                returns, advantages = self._discount_rewards(b, values[b])
                batch_loss = 0
                for transition, ret, advantage in zip(self.transitions[b], returns, advantages):
                    reward, indexes_, outputs_, values_, done_ = transition
                    if done_:
                        continue
                    advantage = advantage.detach()  # Block gradients flow here.
                    probs = F.softmax(outputs_, dim=-1)
                    log_probs = torch.log(probs)
                    log_action_probs = log_probs[indexes_]
                    # log_action_probs = log_probs.gather(1, indexes_.view(batch_size, 1))
                    policy_loss = -log_action_probs * advantage
                    value_loss = (.5 * (values_ - ret) ** 2.)
                    entropy = (-probs * log_probs).sum()
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)
                    advantages.append(advantage)
                    batch_loss = batch_loss + policy_loss + 0.5 * value_loss - 0.0001 * entropy

                    self.batch_stats[b]["mean"]["reward"].append(reward)
                    self.batch_stats[b]["mean"]["policy"].append(policy_loss.item())
                    self.batch_stats[b]["mean"]["value"].append(value_loss.item())
                    self.batch_stats[b]["mean"]["entropy"].append(entropy.item())
                    self.batch_stats[b]["mean"]["confidence"].append(torch.exp(log_action_probs).item())

                if batch_loss != 0:
                    batch_loss_list.append(batch_loss)
                    batch_count += 1
                    self.batch_stats[b]["mean"]["loss"].append(batch_loss.item())
                self.transitions[b] = []
            else:
                # Keep information about transitions for Truncated Backpropagation Through Time.
                # Reward will be set on the next call
                self.transitions[b].append([None, indexes[b], outputs[b], values[b], done[b]])
            self.batch_stats[b]["max"]["score"].append(score_b/infos["game"][b].max_score)

        if len(batch_loss_list) != 0:
            batch_losses = torch.mean(torch.stack(batch_loss_list))
            self.optimizer.zero_grad()
            batch_losses.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.writer.add_scalar('train/loss', batch_losses, self.train_iteration)
            self.writer.add_scalar('train/policy_loss', torch.mean(torch.stack(policy_losses)), self.train_iteration)
            self.writer.add_scalar('train/value_loss', torch.mean(torch.stack(value_losses)), self.train_iteration)
            self.writer.add_scalar('train/entropy', torch.mean(torch.stack(entropies)), self.train_iteration)
            self.writer.add_scalar('train/advantage', torch.mean(torch.stack(advantages)), self.train_iteration)
        
        self.last_score = tuple(last_score)
        self.last_done = done
        if all(done): # Used at the end of the batch to update epsiode stats
            for b in range(batch_size):
                for k, v in self.batch_stats[b]["mean"].items():
                    self.stats["game"][k].append(np.mean(v, axis=0))
                for k, v in self.batch_stats[b]["max"].items():
                    self.stats["game"][k].append(np.max(v, axis=0))

        if self.epsilon > 0.0:
            rand_num = torch.rand((1,),device=self.device) #np.random.uniform(low=0.0, high=1.0, size=(1,))
            less_than_epsilon = (rand_num < self.epsilon).long() # batch
            greater_than_epsilon = 1 - less_than_epsilon
            choosen_idx = less_than_epsilon * sel_rand_action_idx + greater_than_epsilon * sel_action_idx
            action = infos["admissible_commands"][choosen_idx]
        return action

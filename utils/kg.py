import copy
import json
import logging
import random
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
import sng_parser
from torch import sub
from tqdm import tqdm

from utils.amr_parser import AMRSemParser, get_entities, get_triplets
from utils.generic import escape_entities

# Logging formatting
# FORMAT = '%(asctime)s %(message)s'
# logging.basicConfig(format=FORMAT, level='INFO', stream=sys.stdout)
kg = {}
source_paths= defaultdict(dict)


# Constuct World Graph

def shortest_path_subgraph(kg_graph, prev_graph, nodes, inventory_entities=None, command_entities=None, path_len=1, add_all_path=False):
    if inventory_entities is None:
        inventory_entities = []
    if command_entities is None:
        command_entities = []
    # Get non-neighbor nodes: nodes without edges between them
    world_graph = kg_graph.subgraph(list(prev_graph.nodes)+nodes).copy()
    world_graph = nx.compose(prev_graph, world_graph)
    world_graph.remove_edges_from(nx.selfloop_edges(world_graph))

    if path_len < 2:
        return world_graph
    triplets = []
    # Add command related relations
    pruned_entities = list(set(command_entities)-set(inventory_entities))
    if pruned_entities:
        for src_et in inventory_entities:
            for tgt_et in pruned_entities:
                if src_et != tgt_et:
                    try:
                        pair_dist = nx.shortest_path_length(kg_graph, source=src_et, target=tgt_et)
                    except nx.NetworkXNoPath:
                        pair_dist = 0
                    if pair_dist >= 1 and pair_dist <= path_len:
                        triplets.append([src_et, tgt_et, 'relatedTo'])
    else: # no items in the pruned entities, won't happen
        for entities in command_entities:
            for src_et in entities:
                for tgt_et in entities:
                    if src_et != tgt_et:
                        try:
                            pair_dist = nx.shortest_path_length(kg_graph, source=src_et, target=tgt_et)
                        except nx.NetworkXNoPath:
                            pair_dist=0
                        if pair_dist >= 1 and pair_dist <= path_len:
                            triplets.append([src_et, tgt_et, 'relatedTo'])
    world_graph, _= add_triplets_to_graph(world_graph, triplets)
    return world_graph


# Similar Entities
def similar_entity_subgraph(kg_graph, world_graph, entity_list, similar_dict, alias=True, max_simialr_entities=0):
    if len(entity_list) == 0:
        return world_graph
    
    if max_simialr_entities > 0:
        for entity, similar_list in similar_dict.items():
            similar_dict[entity] = similar_list[:max_simialr_entities]
    
    triplets = []
    for entity_1 in entity_list:
        if entity_1 not in similar_dict.keys() or len(similar_dict[entity_1]) <= 0:
            continue
        for entity_2 in entity_list:
            if entity_2 not in similar_dict.keys() or len(similar_dict[entity_2]) <= 0 or entity_1 == entity_2:
                continue
            for similar_entity_info_1 in similar_dict[entity_1]:
                similar_entity_1 = similar_entity_info_1[0]
                is_exist = False
                for similar_entity_info_2 in similar_dict[entity_2]:
                    similar_entity_2 = similar_entity_info_2[0]
                    if kg_graph.has_edge(similar_entity_1, similar_entity_2):
                        if alias:
                            triplets.append([entity_1, entity_2, kg_graph.edges[similar_entity_1, similar_entity_2]['relation']])
                            is_exist = True
                            break
                        elif similar_entity_1 != similar_entity_2:
                            triplets.append([similar_entity_1, similar_entity_2, kg_graph.edges[similar_entity_1, similar_entity_2]['relation']])
                if is_exist:
                    break
    world_graph, _= add_triplets_to_graph(world_graph, triplets)
    world_graph.remove_edges_from(nx.selfloop_edges(world_graph))
        
    return world_graph


# Similar Entities with Pruned by Circumstances
def similar_entity_context_subgraph(kg_graph, world_graph, object_entity_list, container_entity_list, similar_dict, alias=True, max_simialr_entities=0, prune_rate=0, step=0):
    if len(object_entity_list) == 0:
        return world_graph
    
    if max_simialr_entities > 0:
        for entity, similar_list in similar_dict.items():
            similar_dict[entity] = similar_list[:max_simialr_entities]
    
    triplets = []
    for object_entity in tqdm(object_entity_list):
        if object_entity not in similar_dict.keys() or len(similar_dict[object_entity]) <= 0:
            continue
        for container_entity in container_entity_list:
            if container_entity not in similar_dict.keys() or len(similar_dict[container_entity]) <= 0 or object_entity == container_entity:
                continue
            for similar_object_info in similar_dict[object_entity]:
                similar_object = similar_object_info[0]
                is_exist = False
                for similar_container_info in similar_dict[container_entity]:
                    similar_container = similar_container_info[0]
                    if alias:
                        if kg_graph.has_edge(similar_object, similar_container):
                            triplets.append([object_entity, container_entity, kg_graph.edges[similar_object, similar_container]['relation']])
                            is_exist = True
                            break
                    else:
                        if similar_object != similar_container and kg_graph.has_edge(similar_object, similar_container):
                            triplets.append([similar_object, similar_container, kg_graph.edges[similar_object, similar_container]['relation']])
                if is_exist:
                    break
    
    if prune_rate > 0:
        triplets = random.sample(triplets, int(len(triplets) * prune_rate))
    
    # world_graph, _= add_triplets_to_graph(world_graph, triplets, step=step)
    world_graph, _= add_triplets_to_graph(world_graph, triplets)
    world_graph.remove_edges_from(nx.selfloop_edges(world_graph))
                    
    return world_graph


# Constuct KG from triplets

def construct_graph(triplets):
    graph = nx.DiGraph()
    entities = {}
    for [e1, e2, r] in triplets:
        e1 = e1.lower().strip()
        e2 = e2.lower().strip()
        r = r.lower().strip()
        if e1 not in entities:
            graph.add_node(e1)
            entities[e1] = e1
        if e2 not in entities:
            graph.add_node(e2)
            entities[e2] = e2
        # Add Edge information
        if graph.has_edge(e1, e2):
            if r not in graph.edges[e1, e2]['relation']:
                graph.edges[e1, e2]['relation'] += ' ' + r
        else:
            graph.add_edge(e1, e2, relation=r)
    return graph, entities


def construct_graph_amr(triplets):
    graph = nx.DiGraph()
    entities = {}
    for g, step in triplets:
        e1, e2, r = g
        e1 = e1.lower().strip()
        e2 = e2.lower().strip()
        r = r.lower().strip()
        if e1 not in entities:
            graph.add_node(e1)
            entities[e1] = e1
        if e2 not in entities:
            graph.add_node(e2)
            entities[e2] = e2
        # Add Edge information
        if graph.has_edge(e1, e2):
            if r not in graph.edges[e1, e2]['relation']:
                graph.edges[e1, e2]['relation'] += ' ' + r
            graph.edges[e1, e2]['step'] = step
        else:
            graph.add_edge(e1, e2, relation=r, step=step)
    return graph, entities


def construct_graph_vg(triplets):
    graph = nx.DiGraph()
    entities = {}
    edge_count = defaultdict(dict)
    for [e1, e2, r] in triplets:
        e1 = e1.lower().strip()
        e2 = e2.lower().strip()
        r = r.lower().strip()
        if e1 not in entities:
            graph.add_node(e1)
            entities[e1] = e1
        if e2 not in entities:
            graph.add_node(e2)
            entities[e2] = e2
        # Add Edge information
        if graph.has_edge(e1, e2):
            if r in edge_count[e1 + ',' + e2].keys():
                edge_count[e1 + ',' + e2][r] += 1
            else:
                edge_count[e1 + ',' + e2][r] = 1
            relation = max(edge_count[e1 + ',' + e2], key=edge_count[e1 + ',' + e2].get)
            graph.edges[e1, e2]['relation'] = relation
        else:
            graph.add_edge(e1, e2, relation=r)
            edge_count[e1 + ',' + e2][r] = 1
    return graph, entities


def add_triplets_to_graph(graph, triplets, step=None):
    entities = dict(graph.nodes.data())
    for [e1, e2, r] in triplets:
        e1 = e1.lower().strip()
        e2 = e2.lower().strip()
        r = r.lower().strip()
        if e1 not in entities:
            graph.add_node(e1)
            entities[e1] = e1
        if e2 not in entities:
            graph.add_node(e2)
            entities[e2] = e2
        # Add Edge information
        if graph.has_edge(e1, e2):
            if r not in graph.edges[e1, e2]['relation']:
                graph.edges[e1, e2]['relation'] += ' ' + r
            if step is not None:
                graph.edges[e1, e2]['step'] = step
        else:
            if step is not None:
                graph.add_edge(e1, e2, relation=r, step=step)
            else:
                graph.add_edge(e1, e2, relation=r)
    return graph, entities


# Constuct KG from Source file

def construct_kg(filename: str, print_every=1e6, cache_load=True, logger=logging.getLogger(__name__), verbose=True) -> (nx.DiGraph, list, set):
    # access edges with graph.edges.data('relation')
    if 'graph' in kg and cache_load:
        return kg['graph'], kg['triplets'], kg['entities']

    path = Path(filename)
    if not path.exists():
        filename = './kg/conceptnet/kg.txt'

    triplets = []
    with open(filename, 'r') as fp:
        for idx, line in enumerate(fp):
            e1, r, e2  = line.rstrip("\n").rsplit()
            triplets.append([e1.lower().strip(), e2.lower().strip(), r.lower().strip()])
            if idx % print_every == 0 and verbose:
                print("*",end='')
    [graph, entities] = construct_graph(triplets)
    graph = graph.to_undirected(as_view=True) # Since a->b ==> b->a
    if cache_load:
        kg['graph'] = graph
        kg['triplets'] = triplets
        kg['entities'] = entities
    return graph, triplets, entities


def construct_kg_vg(filename: str, print_every=1e5, cache_load=True):
    triplets = []
    with open(filename, 'r') as f:
        rel_data = json.load(f)
    for i, img in enumerate(rel_data):
        for relation in img['relationships']:
            predicate = relation['predicate']
            if "names" in relation['subject'].keys():
                subject = relation['subject']['names'][0]
            else:
                subject = relation['subject']['name']
            if "names" in relation['object'].keys():
                object = relation['object']['names'][0]
            else:
                object = relation['object']['name']
            # predicate = predicate.lower().strip()
            predicate = sentence_preprocess(predicate)
            object = sentence_preprocess(object)
            subject = sentence_preprocess(subject)
            triplets.append([subject, object, predicate])
        if i % print_every == 0:
            print('*',end='')
    graph, entities = construct_graph_vg(triplets)
    graph = graph.to_undirected(as_view=True) # Since a->b ==> b->a
    if cache_load:
        kg['graph'] = graph
        kg['triplets'] = triplets
        kg['entities'] = entities
    return graph, triplets, entities


def draw_graph(graph, title="cleanup", show_relation=True, weights=None, pos=None):
    if not pos:
        pos = nx.spring_layout(graph, k=0.95)
    if weights:
        nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, node_size=1000, node_color=weights.tolist(),
                vmin=np.min(weights), vmax=np.max(weights), node_shape='o', alpha=0.9, font_size=8, with_labels=True,
                label=title,cmap='Blues')
    else:
        nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, node_size=1000, node_color='pink',
                node_shape='o', alpha=0.9, font_size=8, with_labels=True, label=title)
    if show_relation:
        p_edge = nx.draw_networkx_edge_labels(graph, pos, font_size=6, font_color='red',
                                          edge_labels=nx.get_edge_attributes(graph, 'relation'))


def draw_graph_colormap(graph,node_weights, showbar=False, cmap='YlGnBu'):
    # node_weights: maps node id/name to attention weights
    pos = nx.spring_layout(graph, k=0.95)
    weights = []
    for node in graph.nodes:
        weights.append(node_weights[node])
    # cmap = plt.cm.YlGnBu#RdBu
    cmap = plt.get_cmap(cmap)
    vmin = np.min(weights)
    vmax = np.max(weights)
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, node_size=1000,
            node_color=weights, vmin=vmin, vmax=vmax, cmap=cmap,
            node_shape='o', alpha=0.9, font_size=8, with_labels=True, label='Attention')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    if showbar:
        plt.colorbar(sm)
    plt.show()


def sentence_preprocess(phrase):
    table = str.maketrans("", "", string.punctuation)
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
      '½': 'half',
      '—' : '-',
      '™': '',
      '¢': 'cent',
      'ç': 'c',
      'û': 'u',
      'é': 'e',
      '°': ' degree',
      'è': 'e',
      '…': '',
    }
    phrase = phrase.lstrip(' ').rstrip(' ')
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(table)


class RelationExtractor:
    def __init__(self, tokenizer, openie_url="http://localhost:9000/", amr_server_ip='localhost', amr_server_port=None, amr_save_cache='./cache/', amr_load_cache='./cache/amr_cache.pkl'):
        """
        :param tokenizer:
        :param openie_url: server url for Stanford Core NLPOpen IE
        """
        self.tokenizer = tokenizer
        self.openie_url = openie_url
        self.kg_vocab = {}
        self.agent_loc = ''
        self.be_verbs = ['is', 'are']
        self.replace_words = ['make', 'see']
        self.sgp_stop_words = ['what', 'it', 'textworld']
        self.amr_server_ip = amr_server_ip
        self.amr_server_port = amr_server_port
        self.amr_save_cache = amr_save_cache
        self.amr_load_cache = amr_load_cache
        self.amr_replace_relations = ['make-out', 'see', 'look', 'fall', 'notice', 'reveal']
        self.amr_replace_locations = ['there', 'here', 'wall']
        self.object_commands = ['take', 'put']
        self.container_commands = ['examine']
        self.amr_rest = AMRSemParser(amr_server_ip=self.amr_server_ip, amr_server_port=self.amr_server_port, save_cache_folder=self.amr_save_cache, load_cache_file=self.amr_load_cache)

    def call_stanford_openie(self,sentence):
        querystring = {
            "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
            "pipelineLanguage": "en"}
        response = requests.request("POST", self.openie_url, data=sentence, params=querystring)
        response = json.JSONDecoder().decode(response.text)
        return response
    
    def extract_entities_from_commands(self, commands):
        object_list, container_list = [], []
        for cmd in commands:
            if cmd.split()[0] in self.object_commands:
                object_list.append(self.tokenizer.clean_string(cmd))
            if cmd.split()[0] in self.container_commands:
                container_list.append(self.tokenizer.clean_string(cmd))
        object_entities, container_entities = [], []
        for object_cmd in object_list:
            object_entities.append(self.cmd2entity(object_cmd))
        for container_cmd in container_list:
            container_entities.append(self.cmd2entity(container_cmd))
        object_entities = list(set(object_entities))
        container_entities = list(set(container_entities) - set(object_entities))
        return object_entities, container_entities
    
    def cmd2entity(self, command):
        cmd_parts = command.split()
        object_str = ''
        cut_word = ''
        if cmd_parts[0] == 'put' and 'on' in command:
            cut_word = 'on'
        elif cmd_parts[0] == 'take' and 'from' in command:
            cut_word = 'from'
        
        for cmd_part in cmd_parts[1:]:
            if cmd_part == cut_word:
                break
            else:
                object_str += cmd_part + ' '
        return object_str.strip().lower()
    
    def postprocess(self, rawlist_):
        rawlist = []
        for raw in rawlist_:
            raw = raw.strip()
            for r in raw.split('.'):
                if len(r) > 0:
                    rawlist.append(r)
        obslist = []
        for obs in rawlist:
            for o in obs.split(')'):
                o = o.replace('(', '')
                o = o.replace(')', '')
                if len(o) > 0 and o[-1] not in ['.', '?', '!']:
                    o = o + '.'
                if o[-2:] == ' .':
                    o = o[:-2] + '.'
                obs_on = o.split(' on ')
                base_tail = ''
                if len(obs_on) > 1:
                    base_tail = ' on ' + obs_on[-1]
                obs_parts = o.split(',')
                if len(obs_parts) < 2 and o != '':
                    obslist.append(o)
                else:
                    base_head = obs_parts.pop(0).strip()
                    while obs_parts:
                        if len(obs_parts) != 1:
                            obs_part = base_head + ' and ' + obs_parts.pop(0).strip()
                        else:
                            obs_part = base_head + ' ' + obs_parts.pop(0).strip()
                        if base_tail not in obs_part:
                            obs_part = obs_part + base_tail
                        if obs_part[-1] != '.':
                            obs_part += '.'
                        if obs_part != '':
                            obslist.append(obs_part)
        return obslist
    
    def fetch_triplets_amr(self, text, prev_facts, object_entities, container_entities, room):
        current_facts = copy.copy(prev_facts)
        triplets_ = []
        entities = object_entities + container_entities
        if re.search(r'-=\s.+\s=-', text) is not None:
            room = re.search(r'-=\s.+\s=-', text).group().lower()[3:-3]
            triplets_.append(('you', 'location', room))
        obs = self.tokenizer.clean_string(text, preprocess=True)
        obslist_ = [str(sent) for sent in list(self.tokenizer.nlp_eval(obs).sents)]
        obslist = self.postprocess(obslist_)
        triplets, processed_text_list = get_triplets(obslist, self.amr_rest)
        entities.append('floor')
        entities.append('somewhere')
        entities = sorted(entities, key=lambda x: len(x.split(' ')), reverse=True)
        for (s, r, o) in triplets:
            subject, relation, object = s, r, o
            if subject == 'we':
                subject, relation, object = 'you', relation, object
            if relation in self.amr_replace_relations and subject == 'you':
                subject, relation, object = object, 'location', room
            if relation in ['carry', 'pick-up', 'take'] and subject == 'you':
                subject, relation, object = object, 'location', 'you'
            if object in self.amr_replace_locations:
                object = room
            if object in ['ground']:
                object = 'floor'
            
            if 'match' in subject:
                continue
            
            subject_list, object_list = [], []
            if subject == 'you':
                subject_list.append(subject)
            if object == 'you' or object == room:
                object_list.append(object)
            for entity in entities:
                if entity in subject:
                    subject_list.append(entity)
                    subject = subject.replace(entity, '')
                if object != room and entity in object:
                    object_list.append(entity)
                    object = object.replace(entity, '')

            for subj in subject_list:
                for obj in object_list:
                    if subj != obj and obj not in object_entities:
                        triplets_.append((subj, relation, obj))
        
        triplets_.append(('floor', 'location', room))
        
        mod_triplets = {}
        for (s, r, o) in list(set(triplets_)):
            if s in mod_triplets.keys() and mod_triplets[s][1] != room:
                continue
            mod_triplets[s] = [s, o, r]
        
        for s in mod_triplets.keys():
            if mod_triplets[s] == room and s in current_facts.keys() and current_facts[s][1] != room:
                continue
            current_facts[s] = mod_triplets[s]
        
        for rest in list(set(object_entities) - set(current_facts.keys())):
            if rest != 'somewhere':
                current_facts[rest] = [rest, 'somewhere', 'location']
        
        for rest in list(set(container_entities) - set(current_facts.keys())):
            if rest != room:
                current_facts[rest] = [rest, room, 'location']
        
        return current_facts, room, processed_text_list
    
    def fetch_triplets_sgp(self, text, current_graph, prev_action=None):
        obs = self.tokenizer.clean_string(text, preprocess=True)
        doc = self.tokenizer.nlp_eval(obs)
        triplets = []
        room = ''
        for sent in list(doc.sents)[4:]:
            sent = sent.text
            sent = ' '.join(['do' if word in self.be_verbs else word for word in sent.split()])
            sent = ' '.join(['' if word == 'of' else word for word in sent.split()])
            sent = ' '.join([room if word == 'here' else word for word in sent.split()])
            if sent.split()[0] == 'on':
                new_sent = ''
                before_subject = True
                and_object = False
                before_subject_words = []
                for word in sent.split()[:-1]:
                    if word == 'you':
                        before_subject = False
                    if word == 'and':
                        and_object = True
                    if before_subject:
                        before_subject_words.append(word)
                    elif and_object:
                        new_sent += ' '.join(before_subject_words) + '. '
                        and_object = False
                    else:
                        new_sent += word + ' '
                new_sent += ' '.join(before_subject_words)
                sent = new_sent
            
            graphs = sng_parser.parse(sent)
            
            for rel in graphs['relations']:
                subject = graphs['entities'][rel['subject']]['span']
                object = graphs['entities'][rel['object']]['span']
                relation = rel['relation']
                
                is_stop = False
                for stop_word in self.sgp_stop_words:
                    if stop_word in str(subject).lower().split() or stop_word in str(object).lower().split():
                        is_stop = True
                if is_stop:
                    continue
                if subject == 'you' and relation == 'in':
                    room = object
                if subject == 'you' and (relation == 'make' or relation == 'see'):
                    subject = object
                    object = room
                    relation = 'in'
                if 'room' in subject:
                    objcet = subject
                    subject = room
                if 'room' in object:
                    object = room
                    
                triplets.append((subject, relation, object))
                
        for (s, r, o) in triplets:
            current_graph.add_edge(str(s).lower(), str(o).lower(), relation=r.lower())
        return current_graph, triplets
        

    def fetch_triplets(self,text, current_graph, prev_action=None):
        triplets = []
        remove = []
        prev_remove = []
        link = []
        c_id = len(self.kg_vocab.keys())
        obs = self.tokenizer.clean_string(text, preprocess=True)
        dirs = ['north', 'south', 'east', 'west']
        obs = str(obs)
        doc = self.tokenizer.nlp_eval(obs)
        sents = {}
        try:
            sents = self.call_stanford_openie(doc.text)['sentences']
        except:
            print("Error in connecting to Stanford CoreNLP OpenIE Server")
        for ov in sents:
            tokens = ov["tokens"]
            triple = ov['openie']
            for tr in triple:
                h, r, t = tr['subject'].lower(), tr['relation'].lower(), tr['object'].lower()
                if h == 'we':
                    h = 'you'
                    if r == 'are in':
                        r = "'ve entered"

                if h == 'it':
                    break
                triplets.append((h, r, t))

        room = ""
        room_set = False
        for rule in triplets:
            h, r, t = rule
            if 'entered' in r or 'are in' in r or 'walked' in r:
                prev_remove.append(r)
                if not room_set:
                    room = t
                    room_set = True
            if 'should' in r:
                prev_remove.append(r)
            if 'see' in r or 'make out' in r:
                link.append((r, t))
                remove.append(r)
            # else:
            #    link.append((r, t))

        prev_room = self.agent_loc
        self.agent_loc = room
        add_rules = []
        if prev_action is not None:
            for d in dirs:
                if d in prev_action and room != "":
                    add_rules.append((prev_room, d + ' of', room))
        prev_room_subgraph = None
        prev_you_subgraph = None

        for sent in doc.sents:
            sent = sent.text
            if sent == ',' or sent == 'hm .':
                continue
            if 'exit' in sent or 'entranceway' in sent:
                for d in dirs:
                    if d in sent:
                        triplets.append((room, 'has', 'exit to ' + d))
        if prev_room != "":
            graph_copy = current_graph.copy()
            graph_copy.remove_edge('you', prev_room)
            con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)]

            for con_c in con_cs:
                if prev_room in con_c.nodes:
                    prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
                if 'you' in con_c.nodes:
                    prev_you_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)

        for l in link:
            add_rules.append((room, l[0], l[1]))

        for rule in triplets:
            h, r, t = rule
            if r == 'is in':
                if t == 'room':
                    t = room
            if r not in remove:
                add_rules.append((h, r, t))
        edges = list(current_graph.edges)
        for edge in edges:
            r = 'relatedTo'
            if 'relation' in current_graph[edge[0]][edge[1]]:
                r = current_graph[edge[0]][edge[1]]['relation']
            if r in prev_remove:
                current_graph.remove_edge(*edge)

        if prev_you_subgraph is not None:
            current_graph.remove_edges_from(prev_you_subgraph.edges)

        for rule in add_rules:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u != 'it' and u not in self.kg_vocab:
                self.kg_vocab[u] = c_id
                c_id += 1
            if v != 'it' and v not in self.kg_vocab:
                self.kg_vocab[v] = c_id
                c_id += 1
            skip_flag = False
            for skip_token in self.tokenizer.ignore_list:
                if skip_token in u or skip_token in v:
                    skip_flag = True
            if u != 'it' and v != 'it' and not skip_flag:
                r = str(rule[1]).lower()
                if not rule[1] or rule[1] == '':
                    r = 'relatedTo'
                current_graph.add_edge(str(rule[0]).lower(), str(rule[2]).lower(), relation=r)
        prev_edges = current_graph.edges
        if prev_room_subgraph is not None:
            current_graph.add_edges_from(prev_room_subgraph.edges)
        current_edges = current_graph.edges
        return current_graph, add_rules


def khop_neighbor_graph(graph, entities, cutoff=1, max_khop_degree=None):
    all_entities = []
    for et in entities:
        candidates = nx.single_source_shortest_path(graph, et, cutoff=cutoff).keys()
        if not max_khop_degree or len(candidates)<=max_khop_degree:
            all_entities.extend(list(candidates))
    return graph.subgraph(set(entities)|set(all_entities))


def ego_graph_seed_expansion(graph, seed, radius, undirected=True, max_degree=None):
    working_graph = graph
    if undirected:
        working_graph = graph.to_undirected()
    marked = set(seed)
    nodes = set(seed)

    for _ in range(radius):
        border = set()
        for node in marked:
            neighbors = {n for n in working_graph[node]}
            if max_degree is None or len(neighbors) <= max_degree:
                border |= neighbors
        nodes |= border
        marked = border

    return graph.subgraph(nodes)


def shortest_path_seed_expansion(graph, seed, cutoff=None, undirected=True, keep_all=True):
    nodes = set(seed)
    seed = list(seed)

    working_graph = graph
    if undirected:
        working_graph = graph.to_undirected()
    for i in range(len(seed)):
        start = i + 1 if undirected else 0
        for j in range(start, len(seed)):
            try:
                if not keep_all:
                    path = nx.shortest_path(working_graph, seed[i], seed[j])
                    if cutoff is None or len(path) <= cutoff:
                        nodes |= set(path)
                else:
                    paths = nx.all_shortest_paths(working_graph, seed[i], seed[j])
                    for p in paths:
                        if cutoff is None or len(p) <= cutoff:
                            nodes |= set(p)
            except nx.NetworkXNoPath:
                continue
    return graph.subgraph(nodes)


def load_manual_graphs(path, verbose=True):
    path = Path(path)
    manual_world_graphs = {}
    if not path.exists():
        print('None Found.')
        return manual_world_graphs

    files = path.rglob("conceptnet_manual_subgraph-*.tsv")
    for file in files:
        game_id = str(file).split('-')[-1].split('.')[0]
        graph, triplets, entities = construct_kg(file, cache_load=False, verbose=verbose)
        manual_world_graphs[game_id]={}
        manual_world_graphs[game_id]['graph'] = graph
        manual_world_graphs[game_id]['triplets'] = triplets
        manual_world_graphs[game_id]['entities'] = entities
    if verbose:
        print(' DONE')
    return manual_world_graphs




def kg_match(extractor, target_entities, kg_entities):
    result = set()
    kg_entities = escape_entities(kg_entities)
    for e in target_entities:
        e = e.lower().strip()
        result |= extractor(e, kg_entities)
    return result


def save_graph_tsv(graph, path):
    relation_map = nx.get_edge_attributes(graph, 'relation')
    lines = []
    for n1, n2 in graph.edges:
        relations = relation_map[n1, n2].split()
        for r in relations:
            lines.append(f'{n1}\t{r}\t{n2}\n')
    with open(path, 'w') as f:
        f.writelines(lines)

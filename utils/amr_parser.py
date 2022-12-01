import ast
import copy
import os
import pickle
import re
from collections import defaultdict
from logging import Formatter, StreamHandler, getLogger
from pathlib import Path

import requests
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

logger = getLogger("log").getChild("amr")

def get_graph(amr):
    lines = amr.split('\n')
    node_alignments = dict()
    surface_form_nodes = []
    surface_form_indices = [-1, -1]
    edges = []

    for line in lines:
        if line.startswith('# ::tok '):
            line = line.replace('# ::tok ', '')
            tokens = line.split(' ')
        if line.startswith('# ::node'):
            splits = line.split()
            if len(splits) < 4:
                continue
            elif len(splits) == 4:
                node_id = splits[2]
                node_label = splits[3]
                surface_form = node_label
                node_alignments[node_id] = [node_label, surface_form]
            else:
                span_splits = splits[4].split('-')
                node_id = splits[2]
                node_label = splits[3]
                if int(span_splits[0]) >= surface_form_indices[0] \
                        and int(span_splits[1]) <= surface_form_indices[1]:
                    surface_form_nodes.append((node_id, node_label))
                surface_form = \
                    ' '.join(tokens[int(span_splits[0]):int(span_splits[1])])
                node_alignments[node_id] = [node_label, surface_form]
        if line.startswith('# ::edge'):
            splits = line.split()
            edges.append((splits[5], splits[3], splits[6]))
        if line.startswith('# ::short'):
            splits = line.split('\t')
            align_amr_node_id = ast.literal_eval(splits[1])

    return node_alignments, align_amr_node_id, edges


def get_verbnet_preds_from_obslist(obslist,
                                   amr_server_ip='localhost',
                                   amr_server_port=None,
                                   mincount=0, verbose=False,
                                   sem_parser_mode='both',
                                   difficulty='easy'):
    rest_amr = AMRSemParser(amr_server_ip=amr_server_ip,
                            amr_server_port=amr_server_port)
    all_preds = []
    verbnet_facts_logs = {}
    for obs_text in tqdm(obslist):
        verbnet_facts, arity = \
            rest_amr.obs2facts(obs_text,
                               verbose=verbose,
                               mode=sem_parser_mode)
        verbnet_facts_logs[obs_text] = verbnet_facts
        all_preds += list(verbnet_facts.keys())

    rest_amr.save_cache()

    all_preds_set = list(set(all_preds))

    pred_count_dict = {k: all_preds.count(k) for k in all_preds_set}
    all_preds = [k for k, v in pred_count_dict.items() if v > mincount]

    if verbose:
        print('Found {} verbnet preds'.format(len(all_preds)))
        print('Predicates are: ', all_preds)

    return all_preds, pred_count_dict, verbnet_facts_logs


def get_triplets(obslist, rest_amr):
    start_cache_len = len(rest_amr.cache)
    all_triplets = []
    processed_text_list = []
    for obs_text in obslist:
        triplets, processed_text = rest_amr.obs2triplets(obs_text)
        all_triplets += triplets
        processed_text_list.append(processed_text)
    
    if len(rest_amr.cache) > start_cache_len:
        rest_amr.save_cache()

    return all_triplets, processed_text_list


def get_entities(object_list, container_list, rest_amr):
    start_cache_len = len(rest_amr.cache)
    object_entities = []
    container_entities = []
    for t_cmd in object_list:
        entity = rest_amr.cmd2entity(t_cmd)
        object_entities.extend(entity)
    for e_cmd in container_list:
        entity = rest_amr.cmd2entity(e_cmd)
        if entity not in object_entities:
            container_entities.extend(entity)
    
    if len(rest_amr.cache) > start_cache_len:
        rest_amr.save_cache()
    return list(set(object_entities)), list(set(container_entities))


def get_formatted_obs_text(infos):
    obs = infos['description']
    sent_part1 = infos['inventory'].split(':\n')[0]
    sent_part2 = ', '.join(infos['inventory'].split(':\n')[1:])[2:]
    sent = obs.replace('\n', ' ') + ' ' + sent_part1 + ' ' + sent_part2
    return sent


def remove_nextline_space(in_text):
    return re.sub(' +', ' ', in_text)


def detect_joined_noun_phrases(sent, join_token='of', self_assign=False):
    words = sent.split()
    token_dict = {}
    for k, token in enumerate(words):
        if k >= 1 and k < len(words) - 1 and token == join_token:
            full_ent = ' '.join(words[k - 1:k + 2])
            token_dict[words[k - 1]] = full_ent
            token_dict[words[k + 1]] = full_ent
            if self_assign:
                token_dict[full_ent] = full_ent
    return token_dict


def remove_article(s):
    article_list = ['a', 'an', 'the']
    ws = [x for x in s.split() if x not in article_list]
    return ' '.join(ws)


class AMRSemParser:
    def __init__(self,
                 amr_server_ip='localhost',
                 amr_server_port=None,
                 use_amr_cal_str=False,
                 save_cache_folder='./cache/',
                 load_cache_file='./cache/amr_cache.pkl'):
        self.use_amr_cal_str = use_amr_cal_str
        if amr_server_port is None:
            print('AMR is cache only mode')
            self.endpoint = None
        else:
            self.endpoint = \
                'http://%s:%d/verbnet_semantics' % \
                (amr_server_ip, amr_server_port)
        
        self.save_cache_folder = save_cache_folder
        self.save_cache_file = os.path.join(self.save_cache_folder, 'amr_cache.pkl')
        self.load_cache_file = load_cache_file
        if os.path.exists(self.save_cache_file) and Path(self.save_cache_file).stat().st_size > 0:
            self.load_cache_file = self.save_cache_file
        
        self.json_key = 'amr_parse'
        
        self.edge_types = ['be-located-at-91', 'see-01', 'make-out-23', 'carry-01', 'look-01', 'fall-01', 'enter-01', 'reveal-01', 'contain-01', 'notice-01', 'put-01', 'pick-up-04', 'take-01', 'drop-01']
        self.skip_verbs = ['seem', 'appear', 'look']
        self.skip_nouns = ['thing', 'person', 'name', '"Bbq"', '"Hawaiian"']
        self.skip_pairs = {'open-01': 'open', 'close-01': 'close'}
        self.object_verbs = ['insert', 'put', 'take']
        self.container_verbs = ['examine', 'Examine']
        self.skip_words = ['on', '.', 'the', 'a']
        self.bbq_r = re.compile(r'bbq', re.IGNORECASE)
        
        self.nlp = spacy.load('en_core_web_sm')
        self.cache = {}
        self.load_cache()
        self.wrong_cache = {}

    def text2amr(self, text):
        full_ret = {self.json_key: []}
        for sent in sent_tokenize(text):
            if sent in self.cache:
                ret = self.cache[sent]
                full_ret[self.json_key].append(ret)
            else:
                if self.endpoint is None:
                    raise Exception('Need the AMR server for "' + sent + '"')
                r = requests.get(self.endpoint,
                                 params={'text': sent, 'use_coreference': 0})
                ret = r.json()
                if self.json_key not in ret.keys():
                    raise Exception(f'AMR results are failed: {ret}, text: {sent}')
                self.cache[sent] = ret[self.json_key][0]
                full_ret[self.json_key].append(ret[self.json_key][0])

        return full_ret

    def load_cache(self):
        if os.path.exists(self.load_cache_file):
            with open(self.load_cache_file, 'rb') as fp:
                self.cache = pickle.load(fp)
            # print('Loaded cache from ', self.cache_file)
        else:
            self.cache = {}

    def save_cache(self):
        if not os.path.exists(self.save_cache_folder):
            os.mkdir(self.save_cache_folder)
        with open(self.save_cache_file, 'wb') as fp:
            pickle.dump(self.cache, fp)
        # print('Saved cache to ', self.cache_file)
    
    def delete_cache(self, text):
        if any(self.cache):
            wrong_amr = self.cache.pop(text)
            self.wrong_cache[text] = wrong_amr
        else:
            print('amr cache is not existed.')
        
    def search_connection(self, node, edges):
        visit = defaultdict(bool)
        visit[node] = True
        connected_nodes = self.search(node, edges, [], visit)
        return connected_nodes
    
    def search(self, node, edges, connected_nodes, visit):
        for e in edges:
            if e[0] == node and not visit[e[2]]:
                connected_nodes.append(e[2])
                visit[e[2]] = True
                connected_nodes.extend(self.search(e[2], edges, connected_nodes, visit))
        return connected_nodes
    
    def join_adjectives(self, nodes, edges, node_alignments):
        node_str = ''
        nodes.insert(0, '')
        last_add = ''
        for i in range(len(nodes)-1):
            parent, child = nodes[i], nodes[i+1]
            relation = ''
            for e in edges:
                if e[0] == parent and e[2] == child:
                    relation = e[1]
            if relation == 'consist-of' or relation == 'part-of':
                node_str += ' of'
            if node_alignments[child][0] in self.skip_nouns:
                continue
            if node_alignments[child][1] in self.skip_words:
                break
            if last_add != node_alignments[child][1]:
                node_str += f' {node_alignments[child][1]}'
                last_add = node_alignments[child][1]
        if node_str.strip() == 'chest drawers':
            return 'chest of drawers'
        if node_str.strip() == 'bottle cold water':
            return 'bottle of cold water'
        return node_str.strip()

    def amr_preprocess_cmd(self, text):
        text = text.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' ').replace(' some ', ' ').replace('you can ', 'you ')
        text_parts = text.split()
        text_ = ''
        for text_part in text_parts:
            text_ += text_part + ' '
            if text_part in ['take', 'on', 'from']:
                text_ += 'the '
            if text_part in ['put']:
                text_ += 'a '
        text = text_.strip()
        if text.split(' ')[0] not in ['take', 'put']:
            text = text[0].upper() + text[1:]
        text = re.sub(self.bbq_r, 'the BBQ', text_)
        text = text.replace(' tv ', ' TV ').replace('the rotten', 'a rotten')
        return text
    
    def cmd2entity(self, text_):
        text = self.amr_preprocess_cmd(text_)
        is_coffe = False
        if 'coffee' in text:
            text = text.replace('coffee table', 'table')
            is_coffe = True
        
        objects = []
        ret = self.text2amr(text)
        for cnt in range(len(ret[self.json_key])):
            amr_text = ret[self.json_key][cnt]['amr']
            node_alignments, _, edges = get_graph(amr_text)
            
            obj_str = ''
            for e in edges:
                if '-'.join(node_alignments[e[0]][0].split('-')[:-1]) in self.object_verbs and e[1] == 'ARG1':
                    obj = e[2]
                    obj_list = sorted([obj] + self.search_connection(obj, edges), key=lambda x: int(x))
                    obj_str = self.join_adjectives(obj_list, edges, node_alignments)
                
                if '-'.join(node_alignments[e[0]][0].split('-')[:-1]) in self.container_verbs and e[1] == 'ARG1':
                    obj = e[2]
                    obj_list = sorted([obj] + self.search_connection(obj, edges), key=lambda x: int(x))
                    obj_str = self.join_adjectives(obj_list, edges, node_alignments)
            
            if is_coffe and 'table' == obj_str:
                obj_str = 'coffee table'
            
            if obj_str != '':
                objects.append(obj_str.lower())
        return objects
    
    def amr_preprocess_obs(self, text):
        text = text.replace('pair of', '').replace(' a ', ' ').replace(' an ', ' ').replace(' some ', ' the ').replace('you can ', 'you ')
        text_parts = text.split()
        text_ = ''
        is_the = False
        for i, text_part in enumerate(text_parts):
            text_ += text_part + ' '
            if i != len(text_parts) - 1 and text_parts[i+1] != 'the':
                if (text_part in ['see', 'contains'] or (i != 0 and (text_part == 'out' and text_parts[i-1] == 'make') or (text_part == 'is' and text_parts[i-1] == 'there'))):
                    text_ += 'the '
                    is_the = True
                if is_the and text_part == 'and':
                    text_ += 'the '
                if text_part == 'on':
                    text_ += ' the '
        text = remove_nextline_space(' and '.join(text_.split('\n')))
        text = text.replace('you can ', 'you ')
        text = re.sub(self.bbq_r, 'the BBQ', text)
        text = text.replace('the pen', 'a pen').replace('pair of', 'a pair of').replace('the clothes drier', 'a clothes drier').replace('the nightstand', 'a nightstand').replace('the wastepaper', 'a wastepaper').replace('the tv', 'a TV').replace('the dressing table', 'a dressing table').replace('the laundry basket', 'a laundry basket').replace('the sofa', 'a sofa').replace('the toilet roll holder', 'a toilet roll holder').replace('the rotten', 'a rotten').replace('the sugar', 'a sugar').replace('the derby', 'a derby').replace('the checkered tie', 'a checkered tie').replace('the used Q-tip', 'a used Q-tip').replace('the wet', 'a wet').replace('the clean', 'a clean').replace('the cushion', 'a cushion').replace('the brown', 'a brown').replace('the potato peeler', 'a potato peeler').replace('the coffee table', 'a coffee table').replace('the end table', 'an end table').replace('the key', 'a key').replace('the gray', 'a gray').replace('the dirty', 'a dirty').replace('a clean sleeveless shirt', 'clean sleeveless shirt').replace('the soccer', 'a soccer').replace('the patio table', 'a patio table').replace('the bath mat', 'a bath mat')
        text = text.strip()
        text = text[0].upper() + text[1:]
        return text
        
    def obs2triplets(self, text_):
        all_triplets = []
        text = self.amr_preprocess_obs(text_)
        
        if sum([s in text for s in self.skip_verbs]) > 0:
            return all_triplets, text

        ret = self.text2amr(text)
        for cnt in range(len(ret[self.json_key])):
            amr_text = ret[self.json_key][cnt]['amr']
            node_alignments_, _, edges_ = get_graph(amr_text)
            
            is_skip = False
            node_alignments = copy.deepcopy(node_alignments_) # copy
            edges = copy.deepcopy(edges_)
            for edge_ in edges_:
                if edge_[0] == edge_[2]:
                    edges.remove(edge_)
            for n_id, node in node_alignments_.items():
                if node[0] == 'PUNCT' or node[1] == '.' or n_id == 'None':
                    _ = node_alignments.pop(n_id)
                    for edge in edges_:
                        if n_id in edge:
                            edges.remove(edge)
                for tag, word in self.skip_pairs.items():
                    if node[0] == tag and node[1] == word:
                        is_skip = True
            
            if is_skip:
                break
            
            triplets_ = []
            checked_list = [False] * len(edges)
            for i, e in enumerate(edges):
                if e[1] == 'location':
                    obj, loc = [e[0]], e[2]
                    d_edges_obj = [d_e for d_e in edges if d_e[0] == obj[0] and d_e[2] != loc and d_e[1] != 'mode']
                    
                    d_edges_obj_len = len(d_edges_obj)
                    if d_edges_obj_len == 2 and node_alignments[obj[0]][0] == 'and':
                        obj = [e[2] for e in d_edges_obj]
                    elif d_edges_obj_len == 1 or d_edges_obj_len == 0:
                        pass
                    elif d_edges_obj_len >= 2:
                        args = {}
                        for n_e in d_edges_obj:
                            if n_e[1].split('-')[-1] == 'of':
                                continue
                            if 'ARG' in n_e[1]:
                                arg_no = int(re.search(r'[0-9]+', n_e[1]).group())
                                args[arg_no] = n_e
                        args = {arg[0]: arg[1] for arg in sorted(args.items(), key=lambda x:x[0])}
                        if len(args) >= 3:
                            obj = [args[1][2]]
                        elif len(args) == 2:
                            obj = [list(args.values())[1][2]]
                        elif len(args) == 1:
                            obj = [list(args.values())[0][2]]
                        else:
                            pass
                    else:
                        raise ValueError(f'Error: {d_edges_obj}')
                    
                    if len(obj) == 1 and node_alignments[obj[0]][0] == 'and':
                        obj = [d_d_e[2] for d_d_e in edges if d_d_e[0] == obj[0]]
                    
                    for o in obj:
                        triplets_.append((o, e[1], loc))
                
                if checked_list[i]:
                    continue
                for graph_type in self.edge_types:
                    if graph_type == node_alignments[e[0]][0] and 'ARG' in e[1]:
                        sub, obj = [], []
                        d_edges_obj = [e]
                        checked_list[i] = True
                        for j, d_e in enumerate(edges):
                            if d_e[0] == e[0] and 'ARG' in d_e[1] and d_e[1] != e[1]:
                                d_edges_obj.append(d_e)
                                checked_list[j] = True
                        d_edges_obj_len = len(d_edges_obj)
                        if d_edges_obj_len >= 2:
                            args = {}
                            for n_e in d_edges_obj:
                                if len(n_e[1].split('-')) > 1:
                                    continue
                                arg_no = int(re.search(r'[0-9]+', n_e[1]).group())
                                args[arg_no] = n_e
                            args = {arg[0]: arg[1] for arg in sorted(args.items(), key=lambda x:x[0])}
                            if len(args) > 1:
                                sub, obj = [list(args.values())[0][2]], [list(args.values())[1][2]]
                            if graph_type in ['make-out-23', 'put-01', 'drop-01'] and d_edges_obj_len > 2 and len(args) > 2:
                                sub, obj = [list(args.values())[1][2]], [list(args.values())[2][2]]
                        else:
                            continue
                        
                        if len(sub) == 1 and node_alignments[sub[0]][0] == 'and':
                            sub = [d_d_e[2] for d_d_e in edges if d_d_e[0] == sub[0]]
                        if len(obj) == 1 and node_alignments[obj[0]][0] == 'and':
                            obj = [d_d_e[2] for d_d_e in edges if d_d_e[0] == obj[0]]
                        
                        for o in obj:
                            for s in sub:
                                r = node_alignments[e[0]][0]
                                if r == 'be-located-at-91' or r == 'enter-01':
                                    r = 'location'
                                if r == 'contain-01':
                                    r = 'location'
                                    triplets_.append((o, r, s))
                                else:
                                    triplets_.append((s, r, o))
            triplets_adj = [(sorted([sub] + self.search_connection(sub, edges), key=lambda x: int(x)), r, sorted([obj] + self.search_connection(obj, edges), key=lambda x: int(x))) for (sub, r, obj) in triplets_]
            
            triplets = []
            for (sub_adj, r, obj_adj) in triplets_adj:
                sub_str = self.join_adjectives(sub_adj, edges, node_alignments)
                obj_str = self.join_adjectives(obj_adj, edges, node_alignments)
                
                if sub_str == 'oil peanut':
                    sub_str = 'peanut oil'
                if obj_str == 'roll holder toilet':
                    obj_str = 'toilet roll holder'
                sub_str = sub_str.lower()
                obj_str = obj_str.lower()
                
                triplets.append((sub_str, '-'.join(r.split('-')[:-1]) if '-' in r else r, obj_str))
            
            all_triplets += triplets
        
        return all_triplets, text

    def propbank_facts(self, ret,
                       no_use_zero_arg=True,
                       force_single_arity=True,
                       verbose=True, cnt=None):
        facts = {}
        amr_text = ret[self.json_key][cnt]['amr']
        if verbose:
            print('Text:')
            print(ret[self.json_key][cnt]['text'])
        amr_cal_text = ret[self.json_key][cnt]['amr_cal']

        node_alignments, align_amr_node_id, _ = get_graph(amr_text)                
        
        # filter None cases in the keys
        align_amr_node_id = {k: v for k, v in align_amr_node_id.items()
                             if k is not None}
        node_alignments_temp = {int(k): v[1].lower()
                                for k, v in node_alignments.items()
                                if k != 'None'}

        # Disambiguate between different objects
        values = list(node_alignments_temp.values())
        ent_idx = {}
        node_alignments = {}
        for k, v in node_alignments_temp.items():
            count_v = values.count(v)
            if count_v > 1:
                if v in ent_idx:
                    ent_idx[v] -= 1
                else:
                    ent_idx[v] = count_v
                node_alignments[k] = v + '_count_' + str(ent_idx[v])
            else:
                node_alignments[k] = v
        node2surface_mapping = {v: node_alignments[k] for k, v in
                                align_amr_node_id.items()}

        if verbose:
            print('AMR Cal Text:')
            for k, v in ret[self.json_key][cnt].items():
                print(k, ':', v)
        
        node2surface_mapping_ = {}
        mapping_str = ret[self.json_key][cnt]['amr_cal_str'].strip('[]')
        for map_str in re.findall(r'[a-z-0-9]+\([a-z0-9]+\)', mapping_str):
            entity, short = map_str.strip(')').split('(')
            node2surface_mapping_[entity] = short

        pred_args = {}
        pred_values = {}
        modifiers = {}
        cals = defaultdict(dict)
        for item in amr_cal_text:
            pred_name = item['predicate']
            cond = ('-' in pred_name) and ('arg' in pred_name)

            if pred_name.lower() == 'mod':
                obj, mod = item['arguments']
                obj = node2surface_mapping[obj]
                mod = node2surface_mapping[mod]

                if obj not in modifiers:
                    modifiers[obj] = [mod]
                else:
                    modifiers[obj].append(mod)

            if cond:
                verbnet_frame = '-'.join(
                    pred_name.split('.')[0].split('-')[:-1])

                is_neg = item['is_negative']
                if is_neg:
                    verbnet_frame = 'not_' + verbnet_frame

                arg_name = node2surface_mapping[item['arguments'][1]]

                arg_no = int(pred_name.split('.')[-1].split('arg')[-1])
                
                cals[pred_name.split('.')[0]][arg_no] = arg_name

                if arg_no == 0 and no_use_zero_arg:
                    continue

                if verbnet_frame not in pred_args:
                    pred_args[verbnet_frame] = [arg_no]
                    pred_values[verbnet_frame] = [arg_name]
                else:
                    pred_args[verbnet_frame].append(arg_no)
                    pred_values[verbnet_frame].append(arg_name)

        for verb, args in pred_args.items():
            verb_vals = pred_values[verb]
            if verb not in facts:
                facts[verb] = []
            prev_x = -999
            for k, x in enumerate(args):
                argname = pred_values[verb][k]
                mod_argname = argname
                if mod_argname in modifiers:
                    for mod in modifiers[argname]:
                        mod_argname = mod + ' ' + mod_argname
                if x > prev_x and x > 0 and prev_x >= 0:
                    facts[verb][-1].append(mod_argname)
                else:
                    facts[verb].append([mod_argname])
                prev_x = x

        arity = {}
        for k, v in facts.items():
            v_tuple = []
            arity[k] = []
            for item in v:
                arity[k].append(len(item))
                if len(item) > 1:
                    if force_single_arity:
                        v_tuple += item
                    else:
                        v_tuple.append(tuple(item))
                else:
                    v_tuple.append(item[0])
            # v = [tuple(item) for item in v]
            arity[k] = list(set(arity[k]))
            v = list(set(v_tuple))
            facts[k] = v
            
        return facts, arity

    def verbnet_facts(self, ret,
                      no_use_zero_arg=True,
                      force_single_arity=True,
                      verbose=True, cnt=None):
        facts = {}

        res = ret[self.json_key][cnt]
        amr_text = res['amr']

        node_alignments, align_amr_node_id, _ = get_graph(amr_text)
        # filter None cases in the keys
        node_alignments_temp = {int(k): v[1].lower() for k, v in
                                node_alignments.items() if k != 'None'}
        # Disambiguate between different objects
        values = list(node_alignments_temp.values())
        ent_idx = {}
        node_alignments = {}
        for k, v in node_alignments_temp.items():
            count_v = values.count(v)
            if count_v > 1:
                if v in ent_idx:
                    ent_idx[v] -= 1
                else:
                    ent_idx[v] = count_v
                node_alignments[k] = v + '_count_' + str(ent_idx[v])
            else:
                node_alignments[k] = v
        node2surface_mapping = {v: node_alignments[k] for k, v in
                                align_amr_node_id.items()}

        if verbose:
            print('##' * 30)
            print('Grounded smt: ', res['grounded_stmt'])
            print('sem_cal_str: ', res['sem_cal_str'])

        for k, v in res['grounded_stmt'].items():
            verb = k.split('.')[0]
            key_desired = [k_in for k_in in v if verb in k_in]

            if len(key_desired) > 0:
                key_desired = key_desired[0]
            else:
                continue

            for item in v[key_desired][0]:
                pred = item['predicate']
                try:
                    val_facts = tuple([node2surface_mapping[x] for x in
                                       item['arguments'][1:]])
                    if pred in facts:
                        facts[pred].append(val_facts)
                    else:
                        facts[pred] = [val_facts]
                except BaseException:
                    pass
        arity = {}
        for k, v in facts.items():
            v_tuple = []
            arity[k] = []
            for item in v:
                arity[k].append(len(item))
                if len(item) > 1:
                    if force_single_arity:
                        v_tuple += item
                    else:
                        v_tuple.append(tuple(item))
                else:
                    v_tuple.append(item[0])
            arity[k] = list(set(arity[k]))
            v = list(set(v_tuple))
            facts[k] = v
        return facts, arity

    def get_all_possible_adj_nouns(self, phrase):
        list_out = []
        phrase_split = phrase.split()
        for k in range(0, len(phrase_split)):
            list_out.append(' '.join(phrase_split[k:]))
        return list_out

    def get_entity_mappings(self, text, filter_quantifiers, quantifer_words,
                            add_self_mapping=False,
                            add_joined_words=True):

        doc = self.nlp(text)
        list_nps = []
        for nphrase in doc.noun_chunks:
            list_nps.append(remove_article(nphrase.text.lower()))
        list_nps = list(set(list_nps))

        list_nps_dict = {}
        list_root_noun = []
        for x in list_nps:
            if filter_quantifiers:
                x = ' '.join([item for item in x.split()
                              if item not in quantifer_words])
            list_root_noun.append(x.split()[-1])
        ent_idx = {}
        for v in list_nps:
            root_noun = v.split()[-1]
            count_v = list_root_noun.count(root_noun)
            if count_v > 1:
                if root_noun in ent_idx:
                    ent_idx[root_noun] -= 1
                else:
                    ent_idx[root_noun] = count_v
                key = root_noun + '_count_' + str(ent_idx[root_noun])
            else:
                key = root_noun
            list_nps_dict[key] = v
            if add_self_mapping:
                list_nps_dict[v] = v
        if add_joined_words:
            joined_words_dict = detect_joined_noun_phrases(text)
            list_nps_dict = {**list_nps_dict, **joined_words_dict}
        return list_nps_dict

    def obs2facts(self, text, no_use_zero_arg=True, force_single_arity=True,
                  mode='both',
                  verbose=False, filter_nps=True, filter_quantifiers=True):

        text = remove_nextline_space(' and '.join(text.split('\n')))
        ret = self.text2amr(text)
        final_facts = {}
        final_arity = {}

        quantifer_words = ['some ', 'many ', 'lot ', 'few ']
        full_list_nps_dict = self.get_entity_mappings(text, filter_quantifiers,
                                                      quantifer_words)

        for cnt in range(len(ret[self.json_key])):
            if mode == 'both':
                propbank_facts, propbank_arity_facts = \
                    self.propbank_facts(ret,
                                        no_use_zero_arg=no_use_zero_arg,
                                        cnt=cnt,
                                        force_single_arity=force_single_arity,
                                        verbose=False)
                verbnet_facts, verbnet_arity_facts = \
                    self.verbnet_facts(ret,
                                       no_use_zero_arg=no_use_zero_arg,
                                       cnt=cnt,
                                       force_single_arity=force_single_arity,
                                       verbose=False)

                facts = {**verbnet_facts, **propbank_facts}
                arity = {**verbnet_arity_facts, **propbank_arity_facts}
            elif mode == 'verbnet':
                verbnet_facts, verbnet_arity_facts = \
                    self.verbnet_facts(ret,
                                       no_use_zero_arg=no_use_zero_arg,
                                       cnt=cnt,
                                       force_single_arity=force_single_arity,
                                       verbose=False)
                facts = verbnet_facts
                arity = verbnet_arity_facts
            elif mode == 'propbank':
                propbank_facts, propbank_arity_facts =\
                    self.propbank_facts(
                        ret,
                        no_use_zero_arg=no_use_zero_arg,
                        cnt=cnt,
                        force_single_arity=force_single_arity,
                        verbose=False)
                facts = propbank_facts
                arity = propbank_arity_facts
            elif mode == 'none':
                facts = {}
                arity = {}
            else:
                print('Invalid mode. exitting...')
                return None

            # Add handicap in NER based entity linking
            text_sub = ret[self.json_key][cnt]['text']
            if filter_nps:
                list_nps_dict = self.get_entity_mappings(text_sub,
                                                         filter_quantifiers,
                                                         quantifer_words)
                facts_filtered = {}
                for k, v in facts.items():
                    v_filtered = [list_nps_dict[item] for item in v if
                                  item in list_nps_dict]
                    # if not found in single sentence dict search the full text
                    # nps mapping
                    if len(v_filtered) == 0:
                        v_filtered = [full_list_nps_dict[item] for item in v if
                                      item in full_list_nps_dict]
                    if len(v_filtered) > 0:
                        facts_filtered[k] = v_filtered
            else:
                facts_filtered = facts

            for k, v in facts_filtered.items():
                if not (k.startswith('have-')):
                    if k in final_facts:
                        final_facts[k] += v
                    else:
                        final_facts[k] = v

            if verbose:
                print('Text: ', text_sub)
                print('AMR Sem Cal: \n',
                      ret[self.json_key][cnt]['amr_cal_str'])
                print('Facts: \n', facts_filtered)
                print('#' * 50)

            final_arity = {**arity, **final_arity}

        for k, v in final_facts.items():
            all_nouns_adjs = []
            for phrase in v:
                all_nouns_adjs += self.get_all_possible_adj_nouns(phrase)
            all_nouns_adjs = list(set(all_nouns_adjs))
            final_facts[k] = all_nouns_adjs

        return final_facts, final_arity

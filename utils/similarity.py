import argparse
import collections
import os
import csv
import json
import string
import re
import itertools

import numpy as np
import gensim.downloader as api
from tqdm import tqdm


levels = ['easy', 'medium', 'hard']
splits = ['train', 'valid', 'test']
twc_path = "./games/GAMES_100"
vg_path = './vg'
SOURCES_ENTITY = ['twc', 'txt', 'VGOB']
SOURCES_GRAPH = ['VG', 'CN', 'brief', 'complete']

table = str.maketrans("", "", string.punctuation)

save_dir = './similar_dicts'


def sentence_preprocess(phrase):
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
    phrase = phrase.replace('_', ' ')
    return str(phrase).lower().translate(table)


def get_entities(source):
    entities = []
    if source == 'twc':
        game_generation_path = "./game_generation/twc_dataset"
        locations_path = 'twc_locations.json'
        objects_path = 'twc_objects.json'
        doors_path = 'twc_doors.json'
        with open(os.path.join(game_generation_path, locations_path), 'rb') as f:
            twc_locations = json.load(f)
        with open(os.path.join(game_generation_path, objects_path), 'rb') as f:
            twc_objects = json.load(f)
        with open(os.path.join(game_generation_path, doors_path), 'rb') as f:
            twc_doors = json.load(f)
        location_locations = []
        for location in twc_locations.values():
            location_locations.extend(location['locations'])
        doors = []
        for door in twc_doors:
            doors.extend(door['names'])
        twc_entities_set = set(list(twc_objects.keys())) | set(list(twc_locations.keys())) | set(location_locations) | set(doors)
        twc_entities_set_not_door = set(list(twc_objects.keys())) | set(list(twc_locations.keys())) | set(location_locations)
        print('TWC (Objects + Locations + Doors): ', len(twc_entities_set))
        for object in twc_objects.values():
            entities.append(object['entity'])
        return set(entities), twc_entities_set, twc_entities_set_not_door
    elif source == 'txt':
        for level in levels:
            for split in splits:
                with open(os.path.join(twc_path, level, split, 'entities.txt')) as f:
                    file_entities = [l.strip() for l in f.readlines()]
                entities.extend(file_entities)
    elif source == 'add':
        for level in levels:
            for split in splits:
                if split in ['test', 'valid']:
                    for dist in ['in', 'out']:
                        game_dir = os.path.join(twc_path, level, split, dist)
                        game_files = os.listdir(game_dir)
                        for game_file in game_files:
                            if game_file[-4:] == 'json':
                                with open(os.path.join(game_dir, game_file), 'r') as f:
                                    game_json = json.load(f)
                                entities.extend(game_json['metadata']['entities'])
    elif source == 'VGOB':
        with open(os.path.join(vg_path, 'objects.json'), 'r') as f:
            obj_data = json.load(f)
        for img in obj_data:
            for obj in img['objects']:
                for o in obj['names']:
                    entities.append(sentence_preprocess(o))
    entities_set = set(entities)
    print(f'{source} entities: ', len(entities_set))
    return entities_set
    


# get couple (subject and object) from manual_subgraph_brief in commonsense-rl
def get_statistics(source, double=False, alias_dict=None):
    entities = []
    predicates = []
    couples = []
    triplets = []
    if double:
        couples_double = []
        triplets_double = []
        
    def append_statistics(subject, predicate, object):
        if alias_dict is not None:
            if subject in alias_dict.keys():
                subject = alias_dict[subject]
            if object in alias_dict.keys():
                object = alias_dict[object]
        entities.append(subject)
        entities.append(object)
        predicates.append(predicate)
        if len(set([subject, object])) == 1:
            couples.append((subject, object))
        else:
            couples.append(tuple(set([subject, object])))
        triplets.append((subject, predicate, object))
        if double:
            couples_double.append((subject, object))
            couples_double.append((object, subject))
            triplets_double.append((subject, predicate, object))
            triplets_double.append((object, predicate, subject))
            
    if source == 'brief' or source == 'complete':
        for level in levels:
            for split in splits:
                dir_path = os.path.join(twc_path, level, split, f'manual_subgraph_{source}')
                for tsv in os.listdir(dir_path):
                    with open(os.path.join(dir_path, tsv)) as f:
                        for cols in csv.reader(f, delimiter='\t'):
                            if len(cols) == 1:
                                cols = cols[0].split()
                            subject, predicate, object = cols[0], cols[1], cols[2]
                            append_statistics(subject, predicate, object)
    elif source == 'CN':
        for level in levels:
            for split in splits:
                graph_path = os.path.join(twc_path, level, split, 'conceptnet_subgraph.txt')
                with open(graph_path, 'r') as f:
                    for line in f:
                        e1, r, e2  = line.rstrip("\n").rsplit()
                        subject = sentence_preprocess(e1)
                        object = sentence_preprocess(e2)
                        predicate = sentence_preprocess(r)
                        append_statistics(subject, predicate, object)
        # graph_path = os.path.join(twc_path, 'medium', 'test', 'conceptnet_subgraph.txt')
        # with open(graph_path, 'r') as f:
        #     for line in f:
        #         e1, r, e2  = line.rstrip("\n").rsplit()
        #         subject = sentence_preprocess(e1)
        #         object = sentence_preprocess(e2)
        #         predicate = sentence_preprocess(r)
        #         append_statistics(subject, predicate, object)
    elif source == 'VG':
        with open(os.path.join(vg_path, 'relationships.json'), 'r') as f:
            rel_data = json.load(f)
        for img in rel_data:
            for relation in img['relationships']:
                predicate = relation['predicate']
                if "names" in relation['object'].keys():
                    object = relation['object']['names'][0]
                else:
                    object = relation['object']['name']
                if "names" in relation['subject'].keys():
                    subject = relation['subject']['names'][0]
                else:
                    subject = relation['subject']['name']
                predicate = sentence_preprocess(predicate)
                object = sentence_preprocess(object)
                subject = sentence_preprocess(subject)
                append_statistics(subject, predicate, object)
            
    predicates_set = set(predicates)
    couples_set = set(couples)
    entities_set = set(entities)
    triplets_set = set(triplets)
    if double:
        couples_double_set = set(couples_double)
        triplets_double_set = set(triplets_double)
    print(f'{source} entities: ', len(entities_set))
    print(f'{source} predicates: ', len(predicates_set))
    print(f'{source} couples: ', len(couples_set))
    print(f'{source} triplets: ', len(triplets_set))
    if double:
        return entities_set, predicates_set, couples_set, triplets_set, couples_double_set, triplets_double_set
    else:
        return entities_set, predicates_set, couples_set, triplets_set


def make_object_list(entities, list_path='./vg/object_alias_brief.txt'):
    object_list = ''
    for entity in entities:
        object_list += entity + '\n'
    with open(list_path, 'w') as f:
        f.write(object_list)


def preprocess(entity, mode='split', emb_model=None):
    if mode == 'split':
        entity_list = re.split('[_\s]', entity)
    elif mode == 'join':
        entity_list = [''.join(re.split('[_\s]', entity))]
    elif mode == 'exclusive':
        entity_split_list = re.split('[_\s]', entity)
        entity_list = []
        for word in entity_split_list:
            if word in emb_model.key_to_index:
                entity_list.append(word)
    else:
        raise ValueError('This mode is not implemented.')
    return entity_list


def save_json(save_instance, save_path):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_path, 'w') as f:
        json.dump(save_instance, f)


# calculate the similarity of source 1 to source 2
# example: {source_1:'VG', source_2:'brief'}
def similarity_entities(source_1, source_2, emb_model, threshold=0.6, mode='split', is_extend=False, save=None):
    print('=' * 100)
    print(f'Similarity Entity {source_1} - {source_2}')
    if source_1 == 'twc':
        _, _, entities_set_1 = get_entities(source_1)
    elif source_1 in ['txt', 'VGOB', 'add']:
        entities_set_1 = get_entities(source_1)
    else:
        entities_set_1, _, _, _ = get_statistics(source_1)
    if source_2 == 'twc':
        _, _, entities_set_2 = get_entities(source_2)
    elif source_2 in ['txt', 'VGOB', 'add']:
        entities_set_2 = get_entities(source_2)
    else:
        entities_set_2, _, _, _ = get_statistics(source_2)
        
    similar_dict = {}
    unknown_words_1 = []
    unknown_words_2 = []
    similar_dict_len = 0
    for entity_2 in tqdm(list(entities_set_2)):
        similarities = {}
        entity_2_list = preprocess(entity_2, mode='exclusive', emb_model=emb_model)
        if len(entity_2_list) == 0:
            similar_dict[entity_2] = []
            continue
        entity_2_list_list = []
        if is_extend:
            entity_2_root, entity_2_mod_list = entity_2_list[-1], entity_2_list[:-1]
            for i in range(len(entity_2_mod_list) + 1):
                entity_com_list = itertools.combinations(entity_2_mod_list, i)
                for entity_com in entity_com_list:
                    entity_com = list(entity_com)
                    entity_com.append(entity_2_root)
                    entity_2_list_list.append(entity_com)
        else:
            entity_2_list_list.append(entity_2_list)
            
        entity_2_flag = False
        if mode != 'exclusive':
            for word in entity_2_list:
                if word not in emb_model.key_to_index:
                    entity_2_flag = True
        if entity_2_flag:
            unknown_words_2.append(entity_2)
            
        else:
            for entity_2_part in entity_2_list_list:
                for entity_1 in list(entities_set_1):
                    entity_1_list = preprocess(entity_1, mode)
                    entity_1_flag = False
                    for word in entity_1_list:
                        if word not in emb_model.key_to_index:
                            entity_1_flag = True
                    if entity_1_flag:
                        unknown_words_1.append(entity_1)
                        continue
                    similarity = float(emb_model.n_similarity(entity_1_list, entity_2_part))
                    if similarity > threshold:
                        similarities[entity_1] = {'similarity': similarity, 'word': ' '.join(entity_2_part)}
        
        sorted_similarties = sorted(similarities.items(), key=lambda x:x[1]['similarity'], reverse=True)
        similar_dict_len += len(sorted_similarties)
        similar_dict[entity_2] = sorted_similarties
        
    print('-' * 100)
    print(f'{source_1} unknown words: ', len(set(unknown_words_1)))
    print(f'{source_2} unknown words: ', len(set(unknown_words_2)))
    print(f'average number of similar words in {source_1}: ', similar_dict_len / (len(entities_set_2) - len(unknown_words_2)))
    if save:
        extend = 'extend'if is_extend else 'base'
        save_path = os.path.join(save, f'similarity_entity_{source_1}_{source_2}_{threshold}_{extend}.json')
        save_json(similar_dict, save_path)
        print(f"Similar Dicts has been saved in {save_path}")


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


# calculate the similarity of source 1 to source 2
# example: {source_1:'VG', source_2:'brief'}
def similarity_graphs(source_1, source_2, emb_model, threshold=0.65, max_similarities_num=50, mode='split', save=None, predicate=False, predicate_save=None):
    print('=' * 100)
    print(f'Similarity Graph {source_1} - {source_2}')
    _, _, couples_set_1, _, _, triplets_double_set_1 = get_statistics(source_1, double=True)
    _, _, couples_set_2, _ = get_statistics(source_2)
    similar_dict = {}
    similar_couples = []
    unknown_words_1 = []
    unknown_words_2 = []
    similar_dict_len = 0
    negative_list = []
    positive_list = []
    source_2_graph_num = 0
    for (subject_2, object_2) in tqdm(list(couples_set_2)):
        similarities = {}
        subject_2_list = preprocess(subject_2, mode)
        object_2_list = preprocess(object_2, mode)
        source_2_flag = False
        source_2_unknown_word = ''
        for word in subject_2_list + object_2_list:
            if word not in emb_model.key_to_index:
                source_2_flag = True
                source_2_unknown_word = word
                break
        if source_2_flag:
            unknown_words_2.append(source_2_unknown_word)
        else:
            couples_2_list = subject_2_list + object_2_list
            for (subject_1, object_1) in list(couples_set_1):
                subject_1_list = preprocess(subject_1, mode)
                object_1_list = preprocess(object_1, mode)
                source_1_flag = False
                source_1_unknown_word = ''
                for word in subject_1_list + object_1_list:
                    if word not in emb_model.key_to_index:
                        source_1_flag = True
                        source_1_unknown_word = word
                        break
                if source_1_flag:
                    unknown_words_1.append(source_1_unknown_word)
                    continue
                couples_1_list = subject_1_list + object_1_list
                similarity = float(emb_model.n_similarity(couples_1_list, couples_2_list))
                if similarity > threshold:
                    similar_couples.append((subject_1, object_1))
                    similarities[subject_1 + ', ' + object_1] = similarity
        
        sorted_similarities = []
        if len(similarities) > 0:
            sorted_similarities = sorted(similarities.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_similarities) > max_similarities_num:
                sorted_similarities = sorted_similarities[:max_similarities_num]
            positive_list.append([subject_2, object_2])
        else:
            negative_list.append((subject_2, object_2))
        similar_dict_len += len(sorted_similarities)
        similar_dict[subject_2 + ', ' + object_2] = sorted_similarities
        source_2_graph_num += 1
    
    print('-' * 100)
    print(f'{source_1} unknown words: ', len(unknown_words_1))
    print(f'{source_2} unknown words: ', len(unknown_words_2))
    print('average number of similar words in CN: ', similar_dict_len / source_2_graph_num)
    print(f'positive {source_2}: ', len(positive_list))
    print(f'negative {source_2}: ', len(negative_list))
    if save:
        save_path = os.path.join(save, f'similarity_graph_{source_1}_{source_2}.json')
        save_json(similar_dict, save_path)
            
    if predicate:
        predicates = []
        for (subject_1, predicate_1, object_1) in tqdm(list(triplets_double_set_1)):
            if (subject_1, object_1) in list(set(similar_couples)):
                predicates.append(predicate_1)

        c = collections.Counter(predicates)
        most_predicates = c.most_common()
        most_predicates_dict = {x[0]: x[1] for x in most_predicates}
        print(f'{source_1} predicate: ', len(predicates))
        if predicate_save:
            save_path = os.path.join(predicate_save, f'most_predicate_{source_1}_{source_2}.json')
            save_json(most_predicates_dict, save_path)


# calculate the similarity of source 1 to source 2
# example: {source_1:'VG', source_2:'brief'}
def similarity_predicates(source_1, source_2, emb_model, threshold=0.5, max_similarities_num=20, save=None, hand_list=None, hand_weights=None):
    print('=' * 100)
    print(f'Similarity Predicate {source_1} - {source_2}')
    similar_dict = {}
    predicates = collections.defaultdict(list)
    positive_list = []
    negative_list = []
    predicates_lists_1 = []
    predicates_lists_2 = []
    weights_1 = []
    weights_2 = []
    atlocation_list = ['at', 'in', 'on', 'with', 'by', 'beside', 'under', 'over', 'front', 'back', 'above', 'below', 'into', 'out', 'around', 'along']
    hasa_list = ['has']
    if hand_list:
        predicates_lists_1, predicates_lists_2 = hand_list[0], hand_list[1]
        weights_1, weights_2 = hand_weights[0], hand_weights[1]
    if len(predicates_lists_1) == 0:
        _, predicates_set_1, _, _ = get_statistics(source_1)
        for predicate_1 in list(predicates_set_1):
            predicates_lists_1.append(re.split('[_\s]', predicate_1))
    if len(predicates_lists_2) == 0:
        _, predicates_set_2, _, _ = get_statistics(source_2)
        for predicate_2 in list(predicates_set_2):
            predicates_lists_2.append(re.split('[_\s]', predicate_2))
    for i, predicate_list_2 in enumerate(tqdm(predicates_lists_2)):
        source_2_flag = False
        for word in predicate_list_2:
            if word not in emb_model.key_to_index:
                source_2_flag = True
        if source_2_flag:
            continue
        predicate_vector_2 = np.zeros(300)
        for j, predicate_element_2 in enumerate(predicate_list_2):
            if len(weights_2) != 0:
                predicate_vector_2 += weights_2[i][j] * emb_model[predicate_element_2]
            else:
                predicate_vector_2 += (1.0/len(predicate_list_2)) * emb_model[predicate_element_2]
        similarities = {}
        similar_predicates = []
        for predicate_list_1 in predicates_lists_1:
            source_1_flag = False
            for word in predicate_list_1:
                if word not in emb_model.key_to_index:
                    source_1_flag = True
            if source_1_flag:
                continue
            predicate_vector_1 = np.zeros(300)
            for k, predicate_element_1 in enumerate(predicate_list_1):
                if len(weights_1) != 0:
                    predicate_vector_1 += weights_1[i][k] * emb_model[predicate_element_1]
                else:
                    predicate_vector_1 += (1.0/len(predicate_list_1)) * emb_model[predicate_element_1]
            similarity = cos_similarity(predicate_vector_1, predicate_vector_2)
            if similarity > threshold:
                predicate_query_1 = ''.join(predicate_list_1)
                similarities[predicate_query_1] = similarity.item()
                similar_predicates.append(predicate_1)
        
        predicate_query_2 = ''.join(predicate_list_2)
        if predicate_query_2 in atlocation_list:
            predicates['atlocation'].extend(similar_predicates)
        elif predicate_query_2 in hasa_list:
            predicates['hasa'] = similar_predicates
        else:
            predicates[predicate_query_2] = similar_predicates
            
        for key in predicates:
            predicates[key] = list(set(predicates[key]))
        if 'etymologicallyrelatedto' in predicates.keys() and 'relatedto' in predicates.keys():
            predicates['etymologicallyrelatedto'] = predicates['relatedto']

        if len(similarities) > 0:
            sorted_similarities = sorted(similarities.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_similarities) > max_similarities_num:
                sorted_similarities = sorted_similarities[:max_similarities_num]
            positive_list.append(predicate_query_2)
        else:
            negative_list.append(predicate_query_2)
        similar_dict[predicate_query_2] = sorted_similarities
        
    print(positive_list)
    print(negative_list)
        
    if save:
        save_path = os.path.join(save, f'similarity_predicate_{source_1}_{source_2}.json')
        save_json(similar_dict, save_path)
                    
            
# save constrained brief
def construct_constrained_brief(negative_list, save_path):
    for level in levels:
        for split in splits:
            dir_path = os.path.join(twc_path, level, split, 'manual_subgraph_brief')
            save_dir_path = os.path.join(twc_path, level, split, save_path)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            for tsv in os.listdir(dir_path):
                with open(os.path.join(dir_path, tsv)) as f:
                    for cols in csv.reader(f, delimiter='\t'):
                        if len(cols) == 1:
                            cols = cols[0].split()
                        with open(os.path.join(save_dir_path, tsv), 'w') as f_s:
                            writer = csv.writer(f_s, delimiter='\t')
                            if (cols[0], cols[2]) not in negative_list:
                                writer.writerow(cols)


alias_path = './vg/object_alias_brief.txt'
# make alias dictionary
def make_alias(alias_path):
    with open(alias_path, 'r') as f:
        alias_list = [l.strip() for l in f.readlines()]
    alias_dict = {}
    for alias in alias_list:
        candidates = alias.split(',')
        target = candidates[0]
        for o in candidates[1:]:
            alias_dict[o] = target
    return alias_dict


def main(args):
    print('Embedding Model loading ...')
    emb_model = api.load("glove-wiki-gigaword-300")
    print('Embedding Model loaded !')
    similarity_entities('VG', 'add', emb_model, threshold=args.threshold, is_extend=False, save=save_dir)
    # similarity_graphs('VG', 'brief', emb_model, save=save_dir, predicate=True, predicate_save=save_dir)
    # similarity_predicates('VG', 'brief', emb_model, save=save_dir, hand_list=hand_list, hand_weights=hand_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()
    
    main(args)
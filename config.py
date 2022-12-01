import argparse


def model_config():
    parser = argparse.ArgumentParser(add_help=False)

    # general
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--initial_seed', type=int, default=42)
    parser.add_argument('--nruns', type=int, default=5)
    parser.add_argument('--runid', type=int, default=0)
    parser.add_argument('--results_dir', default='results', help='Path to the results files')
    parser.add_argument('--logs_dir', default='./logs', help='Path to the logs files')
    parser.add_argument('--pretrained_model', default='', help='Location of the pretrained command scorer model')
    parser.add_argument('--auto_pretrained_model', action='store_true', help='load the latest model (If this flag is true, pretrained_model is directory.)')
    
    # game
    parser.add_argument('--game_dir', default='./games/twc', help='Location of the game e.g ./games/testbed')
    parser.add_argument('--game_name', help='Name of the game file e.g., kitchen_cleanup_10quest_1.ulx, *.ulx, *.z8')
    parser.add_argument('--dataset_type', default='new', choices=['old', 'new'], help='dataset type')
    parser.add_argument('--difficulty_level', default='easy', choices=['easy','medium', 'hard'],
                        help='difficulty level of the games')    
    # train
    parser.add_argument('--no_train_episodes', type=int, default=100)
    parser.add_argument('--train_max_step_per_episode', type=int, default=50)
    parser.add_argument('--start_ep', type=int, default=0, help='Start no episode in training')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'AdamW'], help='optimizer')
    parser.add_argument('--diff_no_elu', action='store_true', default=False)
    parser.add_argument('--scheduler', action='store_true', default=False, help='whether to use schedule')
    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--train_log', action='store_true', default=False)
    # save model
    parser.add_argument('--specific_save_episodes', nargs='*', type=int, help='--specific_save_episodes 13 25')
    parser.add_argument('--save_model_interval', type=int, default=0)
    
    # validation
    parser.add_argument('--valid', default='none', choices=['none', 'valid', 'test', 'both', 'in', 'out'], help='Validation set')
    parser.add_argument('--valid_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--valid_games_num', type=int, default=5, help='the number of validation games')
    parser.add_argument('--valid_score_interval', type=int, default=10)
    
    # test
    parser.add_argument('--split', default='test', choices=['test', 'valid', 'in', 'out'], help='Whether to run on test or valid')
    parser.add_argument('--no_eval_episodes', type=int, default=5)
    parser.add_argument('--eval_max_step_per_episode', type=int, default=50)
    
    # nlp
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--token_extractor', default='max', help='token extractor: (any or max)')
    parser.add_argument('--corenlp_url', default='http://localhost:9000/',
                        help='URL for Stanford CoreNLP OpenIE Server for the relation extraction for the local graph')
    parser.add_argument('--noun_only_tokens', action='store_true', default=False,
                        help=' Allow only noun for the token extractor')
    parser.add_argument('--use_stopword', action='store_true', default=False,
                        help=' Use stopwords for the token extractor')
    # amr
    parser.add_argument('--amr_server_ip', type=str, default='localhost', help='IP address for AMR server')
    parser.add_argument('--amr_server_port', type=int, default=None, help='PORT num for AMR server')
    parser.add_argument('--amr_save_cache', default='./utils/cache/', type=str, help='AMR save cache directory')
    parser.add_argument('--amr_load_cache', default='./utils/cache/', type=str, help='AMR load cache file path')

    # rl
    parser.add_argument('--hist_scmds_size', type=int, default=3,
                help='Number of recent scored command history to use. Useful when the game has intermediate reward.')
    parser.add_argument('--curiosity', type=float, default=0.0, help='curiosity reward in RL algorithm')
    parser.add_argument('--reward_attenuation',type=float, default=0.0, help='reward attenuation')
    parser.add_argument('--reward_update_frequency', type=int, default=20, help='frequency of model update')
    
    # graph
    parser.add_argument('--agent_type', default='knowledgeaware', choices=['random','simple', 'knowledgeaware'],
                        help='Agent type for the text world: (random, simple, knowledgeable)')
    parser.add_argument('--graph_type', default='', choices=['', 'local', 'world', 'both'],
                        help='What type of graphs to be generated')
    parser.add_argument('--graph_mode', default='evolve', choices=['full', 'evolve'],
                        help='Give Full ground truth graph or evolving knowledge graph: (full, evolve)')
    parser.add_argument('--local_evolve_type', default='direct', choices=['direct', 'ground', 'sgp', 'amr'],
                        help='Type of the generated/evolving strategy for local graph')
    parser.add_argument('--world_evolve_type', default='cdc',
                        choices=['DC', 'CDC', 'NG', 'NG+prune', 'manual', 'EbM', 'EbMNbC', 'goal'],
                        help='Type of the generated/evolving strategy for world graph')
    parser.add_argument('--world_source_type', default='VG', choices=['VG', 'CN'], help='World graph sources: (Visual Genome, ConcepNet)')
    parser.add_argument('--similar_alias', action='store_true', default=False, help='use alias for similar entities')
    parser.add_argument('--similar_dict_type', default='base', choices=['base', 'extend'], help='similar words dictionaly for constructing world graph')
    parser.add_argument('--similarity_threshold', default=0.6, type=float, help='similarity threshold')
    parser.add_argument('--prune_rate', type=float, default=0.0, help='pruning rate for world graph length')
    # Did not use in difference graph
    parser.add_argument('--prune_nodes', action='store_true', default=False,
                        help=' Allow pruning of low-probability nodes in the world-graph')
    parser.add_argument('--prune_start_episode', type=int, default=1, help='Starting the pruning from this episode')
    parser.add_argument('--manual_graph', default='manual_subgraph_brief', help='Manual Graph path in manual mode')
    parser.add_argument('--goal_world', action='store_true', default=False, help='use only world graphs that match goal graphs')

    # network
    parser.add_argument('--batch_size', type=int, default='1', help='Number of the games per batch')
    parser.add_argument('--hidden_size', type=int, default=300, help='num of hidden units for embeddings')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--layer_norm', action='store_true', default=False, help='activate layer normalization')
    parser.add_argument('--nhead', type=int, default=8, help='Head size in Heterogeneous Graph Transformer')
    parser.add_argument('--dropout', action='store_true', default=False, help='add dropout + liner layer')
    parser.add_argument('--diff_network', default='none')
    parser.add_argument('--value_network', default='obs', choices=['obs', 'local', 'diff'], help='value loss network: (observation, local graph, diff graph)')

    # Embeddings
    parser.add_argument('--emb_loc', default='embeddings/', help='Path to the embedding location')
    parser.add_argument('--word_emb_type', default='glove')
    parser.add_argument('--graph_emb_type', help='Knowledge Graph Embedding type for actions: (numberbatch, complex)')
    parser.add_argument('--egreedy_epsilon', type=float, default=0.0, help="Epsilon for the e-greedy exploration")
    parser.add_argument('--truncate', action='store_true', default=False, help='Leave only the last word when including "_"')

    # Debug modes
    parser.add_argument('--graph_extract_mode', action='store_true', default=False, help='activate a mode that extracts world graph used in games')
    parser.add_argument('--skip_cmd', action='store_true', default=False, help='skip commands')

    opt = parser.parse_args()
    
    if opt.graph_type == 'both':
        opt.graph_type = ['local', 'world']
    elif opt.graph_type == 'local':
        opt.graph_type = ['local']
    elif opt.graph_type == 'world':
        opt.graph_type = ['world']
    
    opt.elements = 'seed' + str(opt.initial_seed) + '_runId_' + str(opt.runid)
    
    opt.similar_dict_path = f'./similar_dicts/similarity_entity_{opt.world_source_type}_add_{opt.similarity_threshold}_{opt.similar_dict_type}.json'
    
    return opt

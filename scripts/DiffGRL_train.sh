MODEL_SEED=0
DIFFICULTY=easy
THRESHOLD=0.3
EVOLVE=EbMNbC
HIDDEN=512
SOURCE=VG
OPT=Adam
EPISODES=100
EXP=DiffGRL
AMR_PORT=5000

export PYTHONHASHSEED=$MODEL_SEED
python -u train_agent.py --agent_type knowledgeaware --game_dir ./games/GAMES_100 --game_name *.ulx --difficulty_level $DIFFICULTY --graph_type both --graph_mode evolve --graph_emb_type glove --world_source_type $SOURCE --world_evolve_type $EVOLVE --similar_dict base --similarity_threshold $THRESHOLD --amr_server_ip localhost --amr_server_port $AMR_PORT --nruns 1 --initial_seed $MODEL_SEED --runid $MODEL_SEED --dropout_ratio 0.5 --hidden_size $HIDDEN --curiosity 0.0 --diff_network diffg --value_network diff --exp_name $EXP --no_train_episodes $EPISODES --save_model_interval 10 --valid out --local_evolve_type amr --valid_interval 1 --valid_score_interval 10 --optimizer $OPT --early_stopping 100  --similar_alias

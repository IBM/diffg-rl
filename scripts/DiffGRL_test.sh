MODEL_PATH=$1
MODEL_SEED=0
DIFFICULTY=easy
SPLIT=out
THRESHOLD=0.3
EVOLVE=EbMNbC
HIDDEN=512
SOURCE=VG
EXP=DiffGRL
AMR_PORT=5000

export PYTHONHASHSEED=$MODEL_SEED
SEED=$MODEL_SEED
python -u test_agent.py --agent_type knowledgeaware --game_dir ./games/GAMES_100 --game_name *.ulx --difficulty_level $DIFFICULTY --graph_type both --graph_mode evolve --graph_emb_type glove --world_source_type $SOURCE --world_evolve_type $EVOLVE --similar_dict base --similarity_threshold $THRESHOLD --amr_server_ip localhost --amr_server_port $AMR_PORT --nruns 1 --initial_seed $SEED --runid $SEED --dropout_ratio 0.5 --hidden_size $HIDDEN --diff_network diffg --value_network diff --exp_name TEST_${EXP} --split $SPLIT --local_evolve_type amr --similar_alias --pretrained_model $MODEL_PATH --no_eval_episodes 1

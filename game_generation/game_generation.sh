for level in hard
do
    for split in test
    do
        python twc_make_game.py --objects 7 --rooms 2 --num_games 1 --${split} --output_dir ./game_generation/try/${level}/${split}
    done
done
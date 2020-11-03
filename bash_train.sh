parallel python trainer.py --gamename lunarlander \
--exp_name next_obs_delta \
--num_workers 1 --no_reload --seed {1} ::: {25..27}
# inclusive numbers
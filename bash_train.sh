parallel python trainer.py --gamename lunarlander-sparse \
--implementation UDRL \
--exp_name rerun_adjust_desired_reward --recording_epoch_interval -1 \
--num_workers 1 --seed {1} ::: {25..29}
# inclusive numbers
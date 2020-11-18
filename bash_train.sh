parallel python trainer.py --gamename lunarlander-sparse \
--implementation UDRL \
--exp_name cum_rew_new_init_and_hparams --recording_epoch_interval -1 \
--num_workers 1 --seed {1} ::: {25..29}
# inclusive numbers
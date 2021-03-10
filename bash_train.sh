parallel python trainer.py --gamename lunarlander \
--implementation UDRL \
--exp_name hypernet --recording_epoch_interval -1 \
--num_workers 1 --seed {1} ::: {25..29}
# inclusive numbers
parallel python3 trainer.py --gamename lunarlander \
--exp_name all_desires_test_one \
--num_workers 1 --no_reload --multirun 1 --seed {1} \
--desire_discounted_rew_to_go {2} --desire_cum_rew {3} --desire_horizon {4} \
--desire_state {5} --desire_advantage {6} \
::: 25 27 ::: 0 1 ::: 0 1 ::: 0 1 ::: 0 1 ::: 0 1
# inclusive numbers
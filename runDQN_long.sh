#! /bin/bash
# source .env/bin/activate
tmux new-session -d -s project_experiments
tmux send -t project_experiments "source .env/bin/activate" ENTER
tmux send -t project_experiments "python3 ourgym/long_running_dqn.py" ENTER

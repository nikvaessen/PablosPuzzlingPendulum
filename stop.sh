#! /bin/bash
tmux send-keys C-c -t "project_experiments"
sleep 5
tmux kill-session -t "project_experiments"

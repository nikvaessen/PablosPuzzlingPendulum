#! /bin/bash
sudo apt install -y python3-pip python3-tk tmux
sudo /usr/bin/easy_install3 virtualenv
virtualenv .env -p python3
source .env/bin/activate
pip3 --no-cache-dir install -r requirements.txt
deactivate

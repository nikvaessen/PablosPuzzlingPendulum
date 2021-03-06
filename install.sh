#! /bin/bash
# installing packages
sudo apt install -y python3-pip python3-tk tmux python-pip zip
sudo /usr/bin/easy_install3 virtualenv
pip2 install pydrive backoff

# creating python virtualenv
virtualenv .env -p python3
source .env/bin/activate
pip3 --no-cache-dir install -r requirements.txt
deactivate

# installing gdcp
git clone https://github.com/ctberthiaume/gdcp.git $HOME/gdcp
sudo cp $HOME/gdcp/gdcp /usr/local/bin
cp -r $HOME/PablosPuzzlingPendulum/.gdcp/ $HOME
sed -i -e "s/USER_NAME/$USER/g" $HOME/.gdcp/settings.yaml



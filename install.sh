sudo apt install python3-pip
sudo /usr/bin/easy_install3 virtualenv
virtualenv .env -p python3
source .env/bin/activate
pip install -r requirements.txt
deactivate
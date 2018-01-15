sudo apt install python3-pip
pip install --upgrade virtualenv
virtualenv .env -p python3
source .env/bin/activate
pip install -r requirements.txt
deactivate
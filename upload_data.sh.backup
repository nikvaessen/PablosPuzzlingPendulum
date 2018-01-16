#! /bin/bash
cd $HOME
if [ ! -d "$HOME/SimonsSignificantStatistics" ]; then
    git clone https://github.com/swengeler/SimonsSignificantStatistics.git
fi
cd $HOME/SimonsSignificantStatistics
git pull
if [ ! -d "$HOME/SimonsSignificantStatistics/data" ]; then
    mkdir $HOME/SimonsSignificantStatistics/data
fi
cd $HOME/SimonsSignificantStatistics/data
mv -n $HOME/experiments/*.json $HOME/SimonsSignificantStatistics/data/
git add -A
git commit -m "Data from experiments run by $USER at $(date '+%d/%m/%Y %H:%M:%S')"
git push


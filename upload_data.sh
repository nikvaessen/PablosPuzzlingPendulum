#! /bin/bash
if [ ! -d $HOME/backup ]; then
	mkdir $HOME/backup
fi

if [ ! -d /tmp/ftu ]; then
	mkdir /tmp/ftu
fi

rm /tmp/ftu/*
mv $HOME/experiments/* /tmp/ftu/
cp /tmp/ftu/* $HOME/backup/
zip -r -j /tmp/ftu/experiments_$(echo $USER)_$(date '+%d-%m-%Y_%H:%M:%S').zip /tmp/ftu/*
gdcp upload /tmp/ftu/*.zip -p 10j6F0hFDuuNYzDAKw5GwoFOhAPEMqwxr
rm /tmp/ftu/*

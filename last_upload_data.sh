#! /bin/bash
if [ ! -d $HOME/backup ]; then
	mkdir $HOME/backup
fi

if [ ! -d /tmp/ftu ]; then
	mkdir /tmp/ftu
fi

rm -r /tmp/ftu/*
mv $HOME/experiments*/* /tmp/ftu/
cp -r /tmp/ftu/* $HOME/backup/
zip -r -j /tmp/ftu/experiments_$(echo $USER)_$(date '+%d-%m-%Y_%H:%M:%S').zip /tmp/ftu/*
gdcp upload /tmp/ftu/*.zip -p 1fKfUcfzfpvr_yUnQqc4ZWMv8MYRRA9d-
rm -r /tmp/ftu/*

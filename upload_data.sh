#! /bin/bash
if [ ! -d $HOME/backup ]; then
	mkdir $HOME/backup
fi

if [ ! -d /tmp/ftu/ ]; then
	mkdir /tmp/ftu/
fi

$FILE_NAME_ZIP = "/tmp/ftu/experiments_$USER_$(date '+%d/%m/%Y %H:%M:%S').zip"

rm /tmp/ftu/*
mv $HOME/experiments/* /tmp/ftu/
cp /tmp/ftu/* $HOME/backup/
zip /tmp/ftu* $FILE_NAME_ZIP
gdcp upload $FILE_NAME_ZIP -p 1sE6YuKf4quVl_T6sZqadzq4qBNycY2Ip
rm /tmp/ftu/*

#! /bin/bash
if [ -v 1 ] && [ $1 == -b ] && [ ! -d $HOME/backup ]; then
	mkdir $HOME/backup
fi
for file in $HOME/experiments/*.json; do
    if [ -v 1 ] && [ $1 == -b ]; then
        cp $file $HOME/backup
    fi
    gdcp upload $file -p 1sE6YuKf4quVl_T6sZqadzq4qBNycY2Ip
    rm -f $file
done

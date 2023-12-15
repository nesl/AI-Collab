#!/bin/bash

participants_left=$1

extra=""
if [ ! -z "$2" ]
then
	extra="--server-port $2"
fi


for (( i=1; i<=$1; i++ ))
do
   

   if [[ $i -eq 1 ]]; then
   	COMMAND="gnome-terminal --title robot1 -- bash -ic \""
   else
   	COMMAND+="gnome-terminal --tab --title robot$i -- bash -ic './tutorial_command $extra --robot-number $i;'; sleep 3; "
   fi
done


COMMAND+="./tutorial_command $extra --robot-number 1;\""

echo $COMMAND

eval "$COMMAND"




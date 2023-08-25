#!/bin/bash

PROGRAM="./server_command --robot-number "
COMMAND=""

for (( i=1; i<=$1; i++ ))
do
   if [[ $i -eq 1 ]]; then
   	COMMAND="gnome-terminal --title robot1 -- bash -ic \""
   else
   	COMMAND+="gnome-terminal --tab --title robot$i -- bash -ic './server_command --robot-number $i;'; sleep 2; "
   fi
done

COMMAND+="./server_command --robot-number 1;\""

echo $COMMAND

eval "$COMMAND"




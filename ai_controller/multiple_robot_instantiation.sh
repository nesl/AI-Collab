#!/bin/bash

PROGRAM="./server_command --robot-number "
COMMAND=""

roles=("scout" "scout" "lifter" "lifter" "lifter")
#roles=("general" "general" "general" "general" "general")
#planning=("coordinator" "coordinated" "coordinated" "coordinated" "coordinated")
planning=("equal" "equal" "equal" "equal" "equal")

: '
roles=("scout" "lifter" "general")
role_percentage=(0 0 100)
role_participants=(0 0 0)

participants_left=$1

for (( i=0; i<${#role_percentage[@]}; i++ ))
do
    if [[ $i -eq $((${#role_percentage[@]} - 1)) ]]; then
        role_participants[]

    role_participants[$i]=$(bc <<< "${role_percentage[$i]}/100")
    participants_left=$((participants_left-role_participants[i]))
done

'

for (( i=1; i<=$1; i++ ))
do
   

   if [[ $i -eq 1 ]]; then
   	COMMAND="gnome-terminal --title robot1 -- bash -ic \""
   else
   	COMMAND+="gnome-terminal --tab --title robot$i -- bash -ic './server_command --robot-number $i --role ${roles[$((i-1))]} --planning ${planning[$((i-1))]};'; sleep 3; "
   fi
done

COMMAND+="./server_command --robot-number 1 --role ${roles[0]} --planning ${planning[0]};\""

echo $COMMAND

eval "$COMMAND"




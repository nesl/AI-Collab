#!/bin/bash

array=($(nvidia-xconfig --query-gpu-info | grep -o 'PCI:[0-9].*'))

length=${#array[@]}

for (( j=0; j<length; j++ ));
do
	file_name="xorg-$j.conf"
	nvidia-xconfig --no-xinerama --probe-all-gpus --use-display-device=none --busid="${array[$j]}" -o $file_name 
	echo -e "\n"'Section "ServerFlags"'"\n\t"'Option "AutoAddGPU" "False"'"\nEndSection" >> $file_name
done

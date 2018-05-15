#!/bin/bash -l
source ~/tensorflow/bin/activate
echo 'Echo1'
python node_list_generator.py
echo 'Echo2'
HSTNM="$(hostname | grep -o '[0-9][0-9]')"
echo $HSTNM
STR="$(echo $SLURM_NODELIST | grep -o '[0-9][0-9]')"
echo $STR
if [ ${STR:0:2} == $HSTNM ]
then
   echo 'Parameter Server'
   python a3c_test.py --job_name="ps" --task_index=$HSTNM --number_ps=1
else
   echo 'Worker'
   python a3c_test.py --job_name="worker" --task_index=$HSTNM --number_ps=1
fi

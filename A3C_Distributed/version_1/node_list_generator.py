import os
import re

list = os.environ['SLURM_NODELIST']
pattern_single = re.compile('\d\d')
nodes = pattern_single.findall(list)
pattern_interval = re.compile('\d\d-\d\d')
node_intervals = pattern_interval.findall(list)
if node_intervals:
    for interval in node_intervals:
        start = int(interval[:2])
        end = int(interval[3:])
        if end - start > 1:
            counter = start + 1
            while counter < end:
                nodes.append(counter)
                counter += 1


with open('node_list.txt', 'w') as txt:
    for node in nodes:
        txt.write(str(node) + '\n')

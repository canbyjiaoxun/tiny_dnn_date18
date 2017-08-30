#Extract frequent patterns for first half of operands and then check the hit rate of second half 
#Xun 08/28/17 

from collections import defaultdict 
import operator

#read files 
op_file = open("fully_MUL_op_CNN.txt")
#op_file = open("conv2d_MUL_op_layer1_CNN")
op_list = op_file.readlines() 

#find frequent patterns for the first half  
op_dict = defaultdict(int) 
for op in op_list[:int(0.5*len(op_list))]:
    op_split = op.split()
    if len(op_split) == 2:
       op_tuple = (op_split[0], op_split[1])
       op_dict[op_tuple] += 1 

#assign to a temp dict 
freq_dict = defaultdict(int) 
freq_dict = op_dict 

#check hit rate for the second half 
for op in op_list[int(0.5*len(op_list)):]:
    op_split = op.split()
    if len(op_split) == 2:
       op_tuple = (op_split[0], op_split[1])
       op_dict[op_tuple] += 1 

sorted_freq_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_op_dict = sorted(op_dict.items(), key=operator.itemgetter(1), reverse=True)
print sorted_freq_dict[:20]
print sorted_op_dict[:20]
#print sorted(freq_dict.values(), reverse=True)[:20] 
#print sorted(op_dict.values(), reverse=True)[:20]

    








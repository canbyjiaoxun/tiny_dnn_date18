#profile the most frequent patterns 
#Xun 08/21/17

from collections import defaultdict

size_list = [10, 20, 30, 40, 50] 


op_file = open('test.txt')
#op_file = open('conv2d_MUL_op_layer1_CNN')
op_list = op_file.readlines()
d =defaultdict(int)
for op in op_list:
    op_split = op.split()
    if len(op_split) == 2:
       #op_tuple = (op_split[0], op_split[1])
       op_tuple = (str(round(float(op_split[0]),2)), str(round(float(op_split[1]),2)))
       d[op_tuple] += 1
for size in size_list: 
    print sorted(d.values(), reverse=True)[:size] 
    print "The sum is: ", sum(sorted(d.values(), reverse=True)[:size])
#print sum(sorted(d.values(), reverse=True)[:20])

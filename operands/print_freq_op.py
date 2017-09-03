#print the most frequent patterns 
#Xun 09/2/17


from collections import defaultdict
import struct
import operator

#convert float to IEEE 754
def float_to_bin(value):
    x = sum(ord(b) << 8*i for i,b in enumerate(struct.pack('f', value)))
    return bin(x).replace('0b', '').rjust(32, '0')

#convert IEEE 754 to float 
def bin_to_float(bin_value): 
    f = int(bin_value, 2)
    return struct.unpack('f', struct.pack('I', f))[0]


#define number of bits needed 
bit_chunk = 13

for filename in ["conv2d_MUL_op_layer1_CNN", "conv2d_MUL_op_layer2_CNN", "conv2d_MUL_op_layer3_CNN"]:
#for filename in ["test.txt"]:
     op_file = open(filename)
     freq_op_file = open('freq_op.' + filename + '.' + str(bit_chunk),'w')
     op_list = op_file.readlines()
     d = defaultdict(int)
     for op in op_list:
	 op_split = op.split()
	 try: 
	    op_tuple = (float_to_bin(float(op_split[0]))[0:bit_chunk], float_to_bin(float(op_split[1]))[0:bit_chunk])
	    d[op_tuple] += 1 
	 except ValueError:
	    print "value error: ", op_split 
     sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:50] 
     for i in range(len(sorted_d)):
	 freq_op_file.write(sorted_d[i][0][0] + '\t' + sorted_d[i][0][1] + '\t' + str(bin_to_float(sorted_d[i][0][0]+'1000000000000000000') + bin_to_float(sorted_d[i][0][1]+'1000000000000000000')) + '\n')   
     freq_op_file.close()
     op_file.close()



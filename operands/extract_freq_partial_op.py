#profile the most frequent patterns 
#Xun 08/21/17


from collections import defaultdict
import struct

#convert float to IEEE 754
def float_to_bin(value):
    x = sum(ord(b) << 8*i for i,b in enumerate(struct.pack('f', value)))
    return bin(x).replace('0b', '').rjust(32, '0')
print float_to_bin(-4.5)


#define number of bits needed 
bit_chunk = 20

#op_file = open('fully_MUL_op_CNN.txt')
op_file = open('conv2d_MUL_op_layer1_CNN')
#op_file = open('test.txt')
op_list = op_file.readlines()
d =defaultdict(int)
for op in op_list:
    op_split = op.split()
    try: 
       op_tuple = (float_to_bin(float(op_split[0]))[0:bit_chunk], float_to_bin(float(op_split[1]))[0:bit_chunk])
       d[op_tuple] += 1 
    except ValueError:
       print "value error: ", op_split 
for size in [10, 20, 30, 40, 50]:
    print sorted(d.values(), reverse=True)[:size]
    print "The sum is: ", sum(sorted(d.values(), reverse=True)[:size])



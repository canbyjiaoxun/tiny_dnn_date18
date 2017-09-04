#include <assert.h>
#include <random>
#include <bitset>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <map>
#include <tuple>
namespace tiny_dnn {
namespace kernels {

//layer 1 freq op 
inline std::map<std::tuple<std::string, std::string>, float> BF1(){
    std::map<std::tuple<std::string, std::string>, float> freq_pattern; 
    freq_pattern[std::make_tuple("0000000000000", "1011111110000")] = -1.03125;
    freq_pattern[std::make_tuple("1011111010000", "1011111110000")] = -1.2890625;
    freq_pattern[std::make_tuple("1011111000110", "1011111110000")] = -1.20703125;
    freq_pattern[std::make_tuple("1011111000001", "1011111110000")] = -1.16796875;
    freq_pattern[std::make_tuple("1011110111100", "1011111110000")] = -1.142578125;
    freq_pattern[std::make_tuple("0011111001011", "1011111110000")] = -0.81640625;
    freq_pattern[std::make_tuple("1011111000100", "1011111110000")] = -1.19140625;
    freq_pattern[std::make_tuple("1011111001000", "1011111110000")] = -1.22265625;
    freq_pattern[std::make_tuple("0011110110001", "1011111110000")] = -0.962890625;
    freq_pattern[std::make_tuple("0011110111011", "1011111110000")] = -0.923828125;
    freq_pattern[std::make_tuple("0011111010001", "1011111110000")] = -0.7578125;
    freq_pattern[std::make_tuple("0011110101100", "1011111110000")] = -0.9755859375;
    freq_pattern[std::make_tuple("1011111000101", "1011111110000")] = -1.19921875;
    freq_pattern[std::make_tuple("1011111010010", "1011111110000")] = -1.3203125;
    freq_pattern[std::make_tuple("0011111000001", "1011111110000")] = -0.89453125;
    freq_pattern[std::make_tuple("0011111010011", "1011111110000")] = -0.7265625;
    freq_pattern[std::make_tuple("1011110111000", "1011111110000")] = -1.126953125;
    freq_pattern[std::make_tuple("1011111010110", "1011111110000")] = -1.3828125;
    freq_pattern[std::make_tuple("0011111001001", "1011111110000")] = -0.83203125;
    freq_pattern[std::make_tuple("1011111010011", "1011111110000")] = -1.3359375;
    freq_pattern[std::make_tuple("0011111000100", "1011111110000")] = -0.87109375;
    freq_pattern[std::make_tuple("0011110111000", "1011111110000")] = -0.935546875;
    freq_pattern[std::make_tuple("1011111010001", "1011111110000")] = -1.3046875;
    freq_pattern[std::make_tuple("0011110100010", "1011111110000")] = -0.9951171875;
    freq_pattern[std::make_tuple("0011111000101", "1011111110000")] = -0.86328125;
    freq_pattern[std::make_tuple("0011111001111", "1011111110000")] = -0.78515625;
    freq_pattern[std::make_tuple("0011110110100", "1011111110000")] = -0.951171875;
    freq_pattern[std::make_tuple("0011111000111", "1011111110000")] = -0.84765625;
    freq_pattern[std::make_tuple("1011111001011", "1011111110000")] = -1.24609375;
    freq_pattern[std::make_tuple("0011111010110", "1011111110000")] = -0.6796875;
    freq_pattern[std::make_tuple("1011110100001", "1011111110000")] = -1.0654296875;
    freq_pattern[std::make_tuple("0011111010100", "1011111110000")] = -0.7109375;
    freq_pattern[std::make_tuple("0011111011000", "1011111110000")] = -0.6484375;
    freq_pattern[std::make_tuple("1011110111111", "1011111110000")] = -1.154296875;
    freq_pattern[std::make_tuple("1011110001011", "1011111110000")] = -1.04467773438;
    freq_pattern[std::make_tuple("0011110101010", "1011111110000")] = -0.9794921875;
    freq_pattern[std::make_tuple("0011110110011", "1011111110000")] = -0.955078125;
    freq_pattern[std::make_tuple("0011110011101", "1011111110000")] = -1.00244140625;
    freq_pattern[std::make_tuple("1011110100000", "1011111110000")] = -1.0634765625;
    freq_pattern[std::make_tuple("1011110100010", "1011111110000")] = -1.0673828125;
    freq_pattern[std::make_tuple("0011110101101", "1011111110000")] = -0.9736328125;
    freq_pattern[std::make_tuple("0011110110110", "1011111110000")] = -0.943359375;
    freq_pattern[std::make_tuple("1011111001010", "1011111110000")] = -1.23828125;
    freq_pattern[std::make_tuple("1011111000111", "1011111110000")] = -1.21484375;
    freq_pattern[std::make_tuple("0011110100110", "1011111110000")] = -0.9873046875;
    freq_pattern[std::make_tuple("1011111001111", "1011111110000")] = -1.27734375;
    freq_pattern[std::make_tuple("0011111010111", "1011111110000")] = -0.6640625;
    freq_pattern[std::make_tuple("0011110011110", "1011111110000")] = -1.00146484375;
    freq_pattern[std::make_tuple("0011111001110", "1011111110000")] = -0.79296875;
    freq_pattern[std::make_tuple("1011111001101", "1011111110000")] = -1.26171875;
    return freq_pattern; 
}

//layer 2 freq op
inline std::map<std::tuple<std::string, std::string>, float> BF2(){
    std::map<std::tuple<std::string, std::string>, float> freq_pattern; 
    freq_pattern[std::make_tuple("0000000000000", "0011110111011")] = 0.107421875;
    freq_pattern[std::make_tuple("1011111000000", "0011111100111")] = 0.60546875;
    freq_pattern[std::make_tuple("1011111000000", "0011110111011")] = -0.021484375;
    freq_pattern[std::make_tuple("1011110111011", "0011110111011")] = 0.0;
    freq_pattern[std::make_tuple("0011110110110", "0011111010101")] = 0.423828125;
    freq_pattern[std::make_tuple("0011111000011", "0011111100111")] = 0.88671875;
    freq_pattern[std::make_tuple("0011110110111", "0011111010101")] = 0.427734375;
    freq_pattern[std::make_tuple("0011110111000", "1011110010011")] = 0.07666015625;
    freq_pattern[std::make_tuple("0011110110000", "1011110010011")] = 0.04541015625;
    freq_pattern[std::make_tuple("0011111010001", "1011111001110")] = 0.03515625;
    freq_pattern[std::make_tuple("1011110110100", "0011111100111")] = 0.654296875;
    freq_pattern[std::make_tuple("1011111000000", "0011111011011")] = 0.30078125;
    freq_pattern[std::make_tuple("0011110110111", "0011111100111")] = 0.826171875;
    freq_pattern[std::make_tuple("0011110110000", "0011111100111")] = 0.798828125;
    freq_pattern[std::make_tuple("0011111001001", "0011110111011")] = 0.306640625;
    freq_pattern[std::make_tuple("0011110111010", "0011111100111")] = 0.837890625;
    freq_pattern[std::make_tuple("0011111000100", "0011111100111")] = 0.89453125;
    freq_pattern[std::make_tuple("0011110111001", "0011111100111")] = 0.833984375;
    freq_pattern[std::make_tuple("0011110110000", "0011110111011")] = 0.171875;
    freq_pattern[std::make_tuple("0011110110010", "0011111100111")] = 0.806640625;
    freq_pattern[std::make_tuple("1011111001000", "0011110111011")] = -0.083984375;
    freq_pattern[std::make_tuple("0011110011111", "0011110111011")] = 0.13818359375;
    freq_pattern[std::make_tuple("1011110110010", "0011111100111")] = 0.662109375;
    freq_pattern[std::make_tuple("1011111000001", "1011111001110")] = -0.375;
    freq_pattern[std::make_tuple("0011111000010", "0011111100111")] = 0.87890625;
    freq_pattern[std::make_tuple("1011110100011", "0011111011011")] = 0.3916015625;
    freq_pattern[std::make_tuple("1011110110000", "0011111010101")] = 0.271484375;
    freq_pattern[std::make_tuple("0011110110010", "0011110111011")] = 0.1796875;
    freq_pattern[std::make_tuple("1011110100011", "0011111100111")] = 0.6962890625;
    freq_pattern[std::make_tuple("0011110110000", "1011111001110")] = -0.173828125;
    freq_pattern[std::make_tuple("1011111000010", "0011110111011")] = -0.037109375;
    freq_pattern[std::make_tuple("1011110110000", "1011110010011")] = -0.08349609375;
    freq_pattern[std::make_tuple("1011110110000", "0011110111011")] = 0.04296875;
    freq_pattern[std::make_tuple("1011111000001", "0011111010101")] = 0.19921875;
    freq_pattern[std::make_tuple("0011111000100", "0011110111011")] = 0.267578125;
    freq_pattern[std::make_tuple("1011110110111", "1011111001110")] = -0.330078125;
    freq_pattern[std::make_tuple("0011110110011", "0011111010101")] = 0.412109375;
    freq_pattern[std::make_tuple("0011110110001", "0011111100111")] = 0.802734375;
    freq_pattern[std::make_tuple("1011110101101", "0011111011011")] = 0.3720703125;
    freq_pattern[std::make_tuple("1011110101111", "1011110010011")] = -0.08056640625;
    freq_pattern[std::make_tuple("1011110110110", "0011111011011")] = 0.341796875;
    freq_pattern[std::make_tuple("0011110100101", "0011111010101")] = 0.3779296875;
    freq_pattern[std::make_tuple("0011110110010", "1011110010011")] = 0.05322265625;
    freq_pattern[std::make_tuple("1011111000110", "1011111001110")] = -0.4140625;
    freq_pattern[std::make_tuple("1011111001010", "0011110111011")] = -0.099609375;
    freq_pattern[std::make_tuple("1011111000000", "0011111010101")] = 0.20703125;
    freq_pattern[std::make_tuple("1011111000100", "0011110111011")] = -0.052734375;
    freq_pattern[std::make_tuple("0011111010001", "0011111011011")] = 0.703125;
    freq_pattern[std::make_tuple("0011110101000", "0011111010101")] = 0.3837890625;
    freq_pattern[std::make_tuple("0011111000011", "0011110111011")] = 0.259765625;
    return freq_pattern; 
}

//layer 3 freq op
inline std::map<std::tuple<std::string, std::string>, float> BF3(){
    std::map<std::tuple<std::string, std::string>, float> freq_pattern; 
    freq_pattern[std::make_tuple("0011110110000", "0011111100010")] = 0.642578125;
    freq_pattern[std::make_tuple("1011110110000", "1011110110110")] = -0.15234375;
    freq_pattern[std::make_tuple("0011110110000", "0011111100000")] = 0.580078125;
    freq_pattern[std::make_tuple("0011110110000", "0011111100001")] = 0.611328125;
    freq_pattern[std::make_tuple("1011110110001", "1011110110110")] = -0.15625;
    freq_pattern[std::make_tuple("1011110110010", "0011111100000")] = 0.443359375;
    freq_pattern[std::make_tuple("0011110110010", "1011110110110")] = -0.015625;
    freq_pattern[std::make_tuple("0011110110001", "0011111100000")] = 0.583984375;
    freq_pattern[std::make_tuple("0011110110000", "1011110110110")] = -0.0234375;
    freq_pattern[std::make_tuple("0011110110011", "0011111100000")] = 0.591796875;
    freq_pattern[std::make_tuple("0011110110100", "1011110110110")] = -0.0078125;
    freq_pattern[std::make_tuple("0011110110001", "0011111100010")] = 0.646484375;
    freq_pattern[std::make_tuple("0011110110000", "1011110110101")] = -0.01953125;
    freq_pattern[std::make_tuple("0011110110001", "0011111100001")] = 0.615234375;
    freq_pattern[std::make_tuple("0011110110011", "0011111100010")] = 0.654296875;
    freq_pattern[std::make_tuple("0011110100010", "1011110110110")] = -0.0517578125;
    freq_pattern[std::make_tuple("1011110110001", "1011110110101")] = -0.15234375;
    freq_pattern[std::make_tuple("1011110110000", "1011110110101")] = -0.1484375;
    freq_pattern[std::make_tuple("0011110110101", "0011111100000")] = 0.599609375;
    freq_pattern[std::make_tuple("0011110110001", "1011110110110")] = -0.01953125;
    freq_pattern[std::make_tuple("1011110110010", "0011111100001")] = 0.474609375;
    freq_pattern[std::make_tuple("1011110110000", "1011111000000")] = -0.193359375;
    freq_pattern[std::make_tuple("0011110110011", "0011111100001")] = 0.623046875;
    freq_pattern[std::make_tuple("1011110110000", "1011111100011")] = -0.673828125;
    freq_pattern[std::make_tuple("1011110110001", "1011110110111")] = -0.16015625;
    freq_pattern[std::make_tuple("1011111000001", "0011111010101")] = 0.19921875;
    freq_pattern[std::make_tuple("1011111000000", "1011110110110")] = -0.216796875;
    freq_pattern[std::make_tuple("1011110110000", "1011110110111")] = -0.15625;
    freq_pattern[std::make_tuple("0011110110000", "1011111000001")] = -0.072265625;
    freq_pattern[std::make_tuple("0011110110000", "1011111000000")] = -0.064453125;
    freq_pattern[std::make_tuple("0011110110011", "1011110110110")] = -0.01171875;
    freq_pattern[std::make_tuple("1011110110000", "1011111000001")] = -0.201171875;
    freq_pattern[std::make_tuple("0011110111001", "0011111100000")] = 0.615234375;
    freq_pattern[std::make_tuple("0011110110001", "1011110110101")] = -0.015625;
    freq_pattern[std::make_tuple("0011110110100", "1011111000001")] = -0.056640625;
    freq_pattern[std::make_tuple("1011110110100", "1011111000000")] = -0.208984375;
    freq_pattern[std::make_tuple("0011110100111", "1011110110110")] = -0.0419921875;
    freq_pattern[std::make_tuple("1011111000000", "0011111010101")] = 0.20703125;
    freq_pattern[std::make_tuple("1011110110001", "0011111010101")] = 0.267578125;
    freq_pattern[std::make_tuple("1011110110011", "1011111000000")] = -0.205078125;
    freq_pattern[std::make_tuple("0011111000000", "1011110110110")] = 0.041015625;
    freq_pattern[std::make_tuple("1011110110011", "1011110110110")] = -0.1640625;
    freq_pattern[std::make_tuple("0011110111001", "0011111100010")] = 0.677734375;
    freq_pattern[std::make_tuple("0011110110101", "1011110110110")] = -0.00390625;
    freq_pattern[std::make_tuple("1011110110011", "1011111000001")] = -0.212890625;
    freq_pattern[std::make_tuple("0011110110001", "1011111000000")] = -0.060546875;
    freq_pattern[std::make_tuple("1011110100001", "1011110110110")] = -0.1220703125;
    freq_pattern[std::make_tuple("0011110110010", "1011110110101")] = -0.01171875;
    freq_pattern[std::make_tuple("1011110110100", "0011111100000")] = 0.435546875;
    freq_pattern[std::make_tuple("1011111000000", "1011111000000")] = -0.2578125;
    return freq_pattern; 
}




}
}

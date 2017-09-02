#include <assert.h>
#include <random>
#include <bitset>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <map>
namespace tiny_dnn {
namespace kernels {

//convert float to IEEE 754 string 
inline std::string floatToBinary(float x){
    union
    {
         float input;   // assumes sizeof(float) == sizeof(int)
         int   output;
    }    data;
    data.input = x;
    std::bitset<sizeof(float) * 8>   bits(data.output);
    return bits.to_string(); 
}

//approximate bloom filter hit 
inline float check_BV(float op1, float op2) {
    std::map<std::string, float> freq_pattern;
    freq_pattern[''] = 0.5; 
    std:string op1_str = floatToBinary(op1);
    std:string op2_str = floatToBinary(op2);
    if (freq_pattern[(op1_str, op2_str)].find() != freq_pattern.end())
       return freq_pattern[(op1_str, op2_str)];
       

}







}
}

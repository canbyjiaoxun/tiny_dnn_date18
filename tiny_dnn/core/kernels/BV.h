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
#include "freq_op.h"

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
    std::bitset<sizeof(float) * 8> bits(data.output);
    return bits.to_string(); 
}



//approximate bloom filter hit 
inline float check_BV(float op1, float op2, int layer) {
    std::map<std::tuple<std::string, std::string>, float> freq_pattern;
    if (layer == 1) 
       freq_pattern = BF1(); 
    else if (layer == 2)
       freq_pattern = BF2(); 
    else 
       freq_pattern = BF3(); 
    std::string op1_str = floatToBinary(op1);
    std::string op2_str = floatToBinary(op2);
    //std::cout << "op1: " << op1 << " " << op1_str.substr(0,13) << std::endl; 
    //std::cout << "op2: " << op2 << " " << op2_str.substr(0,13) << std::endl; 
    if (freq_pattern.find(std::make_tuple(op1_str.substr(0,13), op2_str.substr(0,13))) != freq_pattern.end())
    {
       //std::cout << "Hit " << op1 << op2 << std::endl; 
       return freq_pattern[std::make_tuple(op1_str.substr(0,13), op2_str.substr(0,13))]; 
    }  
    else
       return (op1 * op2); 

}



}
}

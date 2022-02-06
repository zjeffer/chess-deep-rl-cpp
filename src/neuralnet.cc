#include "neuralnet.hh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include <tensorflow/c/c_api.h>


NeuralNetwork::NeuralNetwork(){
	// print tensorflow version
	std::cout << "TensorFlow version: " << TF_Version() << std::endl;
}

void NeuralNetwork::predict(std::array<boolBoard, 19> &input, std::array<floatBoard, 73> &output_probs, float &output_value){
	output_probs = {};
	// set output_value to random float between -1 and 1
	output_value = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	// TODO: implement
}
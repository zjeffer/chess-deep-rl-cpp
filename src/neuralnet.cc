#include "neuralnet.hh"
#include "board.hh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

void NeuralNetwork::buildNetwork() {
    // build network
    this->model = Net();
	// move network to device
    this->model.to(this->device);
}

NeuralNetwork::NeuralNetwork() {
    this->buildNetwork();
}

void NeuralNetwork::predict(std::array<boolBoard, 19> &input, std::array<floatBoard, 73> &output_probs, float &output_value) {
    // arrays to tensors
    // torch::Tensor input_tensor = torch::from_blob(input.data(), {1, 19, 8, 8});
	torch::Tensor input_tensor = torch::ones({1, 19, 8, 8});
	input_tensor = input_tensor.to(this->device);

	torch::Tensor output_tensor = this->model.forward(input_tensor);


    printf("Output tensor: \n");
    std::cout << output_tensor << std::endl;
	

    return;

    // TODO: implement
    std::array<float, 4672> probs = {};
    srand((unsigned int)time(NULL));
    for (int i = 0; i < 4672; i++) {
        // random float between 0 and 1
        probs[i] = (float)rand() / (float)RAND_MAX;
    }
    // reshape probs to an array of floatBoards
    for (int i = 0; i < 73; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                output_probs[i].board[j][k] = probs[i * 8 * 8 + j * 8 + k];
            }
        }
    }
    // set output_value to random float between -1 and 1
    output_value = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
}
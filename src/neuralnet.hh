#ifndef neuralnet_hh
#define neuralnet_hh

#include <array>
#include <vector>
#include "board.hh"

class NeuralNetwork {
	public:
		NeuralNetwork();

		void predict(std::array<boolBoard, 19> &input, std::array<floatBoard, 73> &output_probs, float &output_value);
};


#endif // neuralnet_hh
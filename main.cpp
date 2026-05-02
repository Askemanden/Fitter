#include "Model.h"


int main() {

	Model model(2, 2);

	// Input matrix of all combinations of numbers 0 through 9

	// Initialize matrix to correct size
	Matrix x(100, Vector(2));
	// Fill matrix with all combinations of numbers 0 through 9
	for (size_t i = 0; i < 10; ++i) {
		for (size_t j = 0; j < 10; ++j) {
			x[i * 10 + j][0] = i;
			x[i * 10 + j][1] = j;
		}
	}


	Matrix y(100, Vector(2));
	// Fill output matrix with correct values
	for (size_t i = 0; i < 10; ++i) {
		for (size_t j = 0; j < 10; ++j) {
			y[i * 10 + j][0] = i + j;
			y[i * 10 + j][1] = i * 2 + j * 0.5;
		}
	}

	model.train(x, y, 0.0001, 2000);

	// Compare predicted output to true output
	Matrix result = model.predict(x);

	// Print results
	for (size_t i = 0; i < result.size(); ++i) {
		std::cout << "Input: (" << x[i][0] << ", " << x[i][1] << ") ";
		std::cout << "Predicted: (" << result[i][0] << ", " << result[i][1] << ") ";
		std::cout << "True: (" << y[i][0] << ", " << y[i][1] << ")" << std::endl;
	}
	// Print cost
	std::cout << "Cost: " << model.cost(result, y) << std::endl;

	// Print parameters
	std::cout << "Weights: " << std::endl;
	Parameters params = model.getParameters();
	for (size_t i = 0; i < params.weights.size(); ++i) {
		for (size_t j = 0; j < params.weights[i].size(); ++j) {
			std::cout << params.weights[i][j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "Bias: " << std::endl;
	for (size_t i = 0; i < params.bias.size(); ++i) {
		std::cout << params.bias[i] << " ";
	}
	std::cout << std::endl;


	return 0;
}
#include <iostream>
#include "Model.h"

int main()
{
	// Network:
	// 3 inputs
	// 2 hidden layers with 5 neurons each
	// 2 outputs
	Model model({ 3, 5, 5, 2 }, ReLU);

	// Training data
	//
	// Inputs:
	// x0, x1, x2
	//
	// Outputs:
	// y0 = x0 + x1 - x2
	// y1 = x0 * x2
	//
	// Since there are no activation functions,
	// the multiplication target cannot be learned perfectly.
	// But this still tests multi-output training.

	Matrix x = {
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 1},
		{2, 1, 0},
		{2, 2, 1}
	};

	Matrix y = {
		{0, 0}, // 0+0-0 , 0*0
		{-1, 0},
		{1, 0},
		{1, 0},
		{2, 0},
		{0, 1},
		{0, 0},
		{1, 1},
		{3, 0},
		{3, 2}
	};

	std::cout << "Predictions before training:\n\n";

	Matrix predBefore = model.predictAll(x);

	for (size_t i = 0; i < x.size(); ++i) {

		std::cout << "Input: ";

		for (double v : x[i]) {
			std::cout << v << " ";
		}

		std::cout << " -> Predicted: ";

		for (double v : predBefore[i]) {
			std::cout << v << " ";
		}

		std::cout << "\n";
	}

	std::cout << "\nTraining...\n\n";

	model.train(x, y, 0.01, 3000);

	std::cout << "Predictions after training:\n\n";

	Matrix predAfter = model.predictAll(x);

	for (size_t i = 0; i < x.size(); ++i) {

		std::cout << "Input: ";

		for (double v : x[i]) {
			std::cout << v << " ";
		}

		std::cout << " -> Predicted: ";

		for (double v : predAfter[i]) {
			std::cout << v << " ";
		}

		std::cout << " | Expected: ";

		for (double v : y[i]) {
			std::cout << v << " ";
		}

		std::cout << "\n";
	}

	return 0;
}
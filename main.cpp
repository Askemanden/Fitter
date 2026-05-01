#include "Model.h"


int main() {

	Model model(2, 1);

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

	// Output matrix of number 1 times 6 plus number two times 1.5
	Matrix y(100, Vector(1));
	// Fill output matrix with correct values
	for (size_t i = 0; i < 10; ++i) {
		for (size_t j = 0; j < 10; ++j) {
			y[i * 10 + j][0] = i * 6 + j * 1.5;
		}
	}


	model.train(x, y, 0.0001, 10000);

	// Compare predicted output to true output
	Matrix result = model.predict(x);

	// Print results
	for (size_t i = 0; i < 100; ++i) {
		std::cout << "Input: (" << x[i][0] << ", " << x[i][1] << ") ";
		std::cout << "Predicted: " << result[i][0] << " ";
		std::cout << "True: " << y[i][0] << std::endl;
	}


	return 0;
}
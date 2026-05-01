#include <iostream>
#include <vector>

typedef std::vector<std::vector<double>> Matrix;
typedef double (*CostFunction)(std::vector<double>, std::vector<double>);
typedef std::vector<double>(*PredictorFunction)(Matrix, std::vector<double>);

typedef struct TrainingSet{
	Matrix x;
	std::vector<double> y;
} TrainingSet;


static double cost(std::vector<double> y_predict, std::vector<double> y) {

	double result = 0.0;

	for (size_t i = 0; i < y.size(); ++i) {
		result += (y_predict[i] - y[i]) * (y_predict[i] - y[i]);
	}

	return result / y.size();
}

static std::vector<double> predict(Matrix data, std::vector<double> params) {
	std::vector<double> result;
	for (size_t i = 0; i < data.size(); ++i) {
		double prediction = 0.0;
		for (size_t j = 0; j < data[i].size(); ++j) {
			prediction += data[i][j] * params[j];
		}
		result.push_back(prediction);
	}
	return result;
}


/**
* @brief Computes the numerical gradient of the cost function with respect to the parameters. Formula (L(f(x + h),y) - L(f(x - h),y)) / (2 * h)
* @param h The step size for numerical differentiation.
* @param data The input data matrix. Each row represents a data point, and each column represents a feature.
* @param params The current parameters of the model. This is a vector where each element corresponds to a parameter that we want to optimize.
* @param y The actual output values corresponding to the input data.
* @param predictor The function used to make predictions based on the input data and parameters.
* @param cost The cost function used to evaluate the difference between predicted and actual values.
* @return A vector representing the gradient of the cost function with respect to each parameter.
*/
static std::vector<double> gradient(double h, Matrix data, std::vector<double> params, std::vector<double> y, PredictorFunction predictor, CostFunction cost) {
	std::vector<double> gradient(params.size(), 0.0);
	
	for (size_t j = 0; j < params.size(); ++j) {
		std::vector<double> stepped_params_up = params;
		stepped_params_up[j] += h;
		std::vector<double> stepped_params_down = params;
		stepped_params_down[j] -= h;


		double slope = (cost(predictor(data, stepped_params_up), y) - cost(predictor(data, stepped_params_down), y)) / (2 * h);
		gradient[j] = slope;
	}
	return gradient;
}

Matrix data = {
	{1.0, 1.0},
	{1.0, 2.0},
	{1.0, 3.0},
	{1.0, 4.0},
	{1.0, 5.0},
	{2.0, 1.0},
	{2.0, 2.0},
	{2.0, 3.0},
	{2.0, 4.0},
	{2.0, 5.0},
	{3.0, 1.0},
	{3.0, 2.0},
	{3.0, 3.0},
	{3.0, 4.0},
	{3.0, 5.0},
	{4.0, 1.0},
	{4.0, 2.0},
	{4.0, 3.0},
	{4.0, 4.0},
	{4.0, 5.0},
};

// = data[i][0] + 2 * data[i][1]
std::vector<double> y = { 
	3.0, 5.0, 7.0, 9.0, 11.0,
	4.0, 6.0, 8.0, 10.0, 12.0,
	5.0, 7.0, 9.0, 11.0, 13.0,
	6.0, 8.0, 10.0, 12.0, 14.0
};

std::vector<double> params = { 0.0, 0.0 };

int main() {
	for (size_t i = 0; i < 20000; ++i) {

		std::vector<double> y_predict = predict(data, params);
		double current_cost = cost(y_predict, y);

		if (i % 400 == 0) {
			std::cout << "Current cost: " << current_cost << std::endl;
			std::cout << "Predictions | Actual" << std::endl;
			for (size_t i = 0; i < y_predict.size(); ++i) {
				std::cout << y_predict[i] << " | " << y[i] << std::endl;
			}
			std::cout << std::endl;
		}

		// Compute the gradient
		std::vector<double> grad = gradient(0.001, data, params, y, predict, cost);
		if (i % 400 == 0) {
			std::cout << "Gradient: ";
			for (size_t i = 0; i < grad.size(); ++i) {
				std::cout << grad[i] << " | ";
			}
			std::cout << std::endl;
		}

		// Update parameters

		for (size_t j = 0; j < params.size(); ++j) {
			params[j] -= 0.0001 * grad[j];
		}

		if(i%400 ==0) {
			std::cout << "Updated parameters: ";
			for (size_t j = 0; j < params.size(); ++j) {
				std::cout << params[j] << " | ";
			}
			std::cout << std::endl;
		}

	}


	return 0;
}
#include "Model.h"



double squareError(const Matrix& y_pred, const Matrix& y_true)
{
	if (y_pred.empty() || y_true.empty() ||
		y_pred[0].empty() || y_true[0].empty()) {
		throw std::invalid_argument("Matrices cannot be empty.");
	}

	if (y_pred.size() != y_true.size() ||
		y_pred[0].size() != y_true[0].size()) {
		throw std::invalid_argument("Size mismatch.");
	}

	double result = 0.0;
	for (size_t i = 0; i < y_true.size(); ++i) {
		for (size_t j = 0; j < y_true[i].size(); ++j) {
			result += (y_pred[i][j] - y_true[i][j]) * (y_pred[i][j] - y_true[i][j]);
		}
	}
	return result / (y_true.size() * y_true[0].size());
}

Vector ReLU(const Vector& input)
{
	Vector result = input;
	for (double& v : result) {
		if (v < 0) {
			v = 0;
		}
	}
	return result;
}
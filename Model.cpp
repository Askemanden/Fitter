#include "Model.h"

Vector Model::predict(const Vector& x, const Parameters& params) const
{
	if (x.size() != m_inputSize) {
		throw std::invalid_argument("Input size does not match model input size.");
	}

	Vector result(m_outputSize, 0.0);

	for (size_t i = 0; i < m_outputSize; ++i) {
		double prediction = 0.0;
		for (size_t j = 0; j < m_inputSize; ++j) {
			prediction += x[j] * params.weights[i][j];
		}
		prediction += params.bias[i];
		result[i] = prediction;
	}

	return result;
}

Vector Model::predict(const Vector& x) const
{
	return predict(x, m_parameters);
}

Matrix Model::predict(const Matrix& x, const Parameters& params) const
{ 
	if (x.empty() || x[0].size() != m_inputSize) {
		throw std::invalid_argument("Input size does not match model input size.");
	}
	Matrix result(x.size(), Vector(m_outputSize, 0.0));
	for (size_t i = 0; i < x.size(); ++i) {
		result[i] = predict(x[i], params);
	}
	return result;
}

Matrix Model::predict(const Matrix& x) const
{
	return predict(x, m_parameters);
}

double Model::cost(const Matrix& y_pred, const Matrix& y_true)
{
	if (y_pred.size() != y_true.size() || y_pred[0].size() != y_true[0].size() ) {
		throw std::invalid_argument("Predicted and true output sizes do not match.");
	}
	else if (y_pred.empty() || y_pred[0].empty()) {
		throw std::invalid_argument("Predicted and true output cannot be empty.");
	}

	double result = 0.0;
	for (size_t i = 0; i < y_true.size(); ++i) {
		for (size_t j = 0; j < y_true[i].size(); ++j) {
			result += (y_pred[i][j] - y_true[i][j]) * (y_pred[i][j] - y_true[i][j]);
		}
	}
	return result / (y_true.size() * y_true[0].size());
}

Parameters Model::gradient(double h, const Matrix& x, const Matrix& y)
{
	Parameters grad;
	grad.weights = Matrix(m_outputSize, Vector(m_inputSize, 0.0));
	grad.bias = Vector(m_outputSize, 0.0);
	for (size_t i = 0; i < m_outputSize; ++i) {
		for (size_t j = 0; j < m_inputSize; ++j) {
			Parameters stepped_params_up = m_parameters;
			stepped_params_up.weights[i][j] += h;
			Parameters stepped_params_down = m_parameters;
			stepped_params_down.weights[i][j] -= h;
			double slope = (cost(predict(x, stepped_params_up), y) - cost(predict(x, stepped_params_down), y)) / (2 * h);
			grad.weights[i][j] = slope;
		}
	}
	for (size_t i = 0; i < m_outputSize; ++i) {
		Parameters stepped_params_up = m_parameters;
		stepped_params_up.bias[i] += h;
		Parameters stepped_params_down = m_parameters;
		stepped_params_down.bias[i] -= h;
		double slope = (cost(predict(x, stepped_params_up), y) - cost(predict(x, stepped_params_down), y)) / (2 * h);
		grad.bias[i] = slope;
	}
	return grad;
}

void Model::train(const Matrix& x, const Matrix& y, double learningRate, size_t max_iterations)
{
	for (size_t iter = 0; iter < max_iterations; ++iter) {
		Parameters grad = gradient(1e-5, x, y);
		for (size_t i = 0; i < m_outputSize; ++i) {
			for (size_t j = 0; j < m_inputSize; ++j) {
				m_parameters.weights[i][j] -= learningRate * grad.weights[i][j];
			}
			m_parameters.bias[i] -= learningRate * grad.bias[i];
		}
	}

}

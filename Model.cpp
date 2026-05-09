#include "Model.h"


Vector Model::computeLayer(const Vector& input, size_t layerIndex) const
{

	const Parameters& params = m_parameters[layerIndex];
	// Layers start at hidden layer 1, shape starts at input size. 
	// Therefore we offset layerIndex by 1 to get the correct input and output sizes. 
	size_t inputSize = m_shape[layerIndex];
	size_t outputSize = m_shape[layerIndex + 1];

	if (input.size() != inputSize) {
		throw std::invalid_argument("Input size does not match model input size.");
	}

	Vector result(outputSize, 0.0);

	for (size_t i = 0; i < outputSize; ++i) {
		double prediction = 0.0;
		for (size_t j = 0; j < inputSize; ++j) {
			prediction += input[j] * params.weights[i][j];
		}
		prediction += params.bias[i];
		result[i] = prediction;
	}

	return result;
}

Vector Model::ReLU(const Vector& input) const
{
	Vector result = input;
	for (double& v : result) {
		if (v < 0) {
			v = 0;
		}
	}
	return result;
}

Vector Model::predict(const Vector& x) const
{
	Vector y = x;
	for (size_t i = 0; i < m_parameters.size(); ++i) {
		y = computeLayer(y, i);
		if (i < m_parameters.size() - 1) { // Don't apply ReLU to output layer
			y = ReLU(y);
		}
	}
	return y;
}

Matrix Model::predictAll(const Matrix& x) const
{
	for (const auto& row : x) {
		if (row.size() != m_shape[0]) {
			throw std::invalid_argument("Input size does not match model input size.");
		}
	}

	Matrix result;

	for (const auto& row : x) {
		result.push_back(predict(row));
	}

	return result;
}

double Model::cost(const Matrix& y_pred, const Matrix& y_true)
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

std::vector<Parameters> Model::gradient(double h, const Matrix& x, const Matrix& y)
{
	std::vector<Parameters> grad;

	for (const auto& p : m_parameters) {
		grad.emplace_back(p.weights.size(), p.weights[0].size());
	}

	for (size_t i = 0; i < m_parameters.size(); ++i) {
		for (size_t j = 0; j < m_parameters[i].weights.size(); ++j) {
			for (size_t k = 0; k < m_parameters[i].weights[j].size(); ++k) {

				// Step parameters down by h and calculate cost
				m_parameters[i].weights[j][k] -= h;
				double costDown = cost(predictAll(x), y);

				// Step parameters up by 2h, 1 to compensate for last step and one to step up,
				// and calculate cost
				m_parameters[i].weights[j][k] += 2 * h;
				double costUp = cost(predictAll(x), y);

				grad[i].weights[j][k] = (costUp - costDown) / (2 * h);

				m_parameters[i].weights[j][k] -= h; // Reset to original value
			}
		}
		for (size_t j = 0; j < m_parameters[i].bias.size(); ++j) {
			// Step bias down by h and calculate cost
			m_parameters[i].bias[j] -= h;
			double costDown = cost(predictAll(x), y);

			// Step bias up by 2h, 1 to compensate for last step and one to step up,
			// and calculate cost
			m_parameters[i].bias[j] += 2 * h;
			double costUp = cost(predictAll(x), y);

			grad[i].bias[j] = (costUp - costDown) / (2 * h);

			m_parameters[i].bias[j] -= h; // Reset to original value
		}
	}
	return grad;
}

void Model::train(const Matrix& x, const Matrix& y, double learningRate, size_t max_iterations)
{
	for (size_t i = 0; i < max_iterations; ++i) {
		std::vector<Parameters> grad = gradient(1e-5, x, y);

		for (size_t j = 0; j < m_parameters.size(); ++j) {

			for (size_t k = 0; k < m_parameters[j].weights.size(); ++k) {

				for (size_t l = 0; l < m_parameters[j].weights[k].size(); ++l) {

					m_parameters[j].weights[k][l] -= learningRate * grad[j].weights[k][l];
				}
			}

			for (size_t k = 0; k < m_parameters[j].bias.size(); ++k) {

				m_parameters[j].bias[k] -= learningRate * grad[j].bias[k];
			}
		}
	}

}

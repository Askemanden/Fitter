#pragma once
#include <random>
#include <initializer_list>
#include "ActivationFunction.h"
#include "Vector.h"

struct Parameters {
	Matrix weights;
	Vector bias;

	// Random initialization
	Parameters(size_t outSize, size_t inSize)
		: weights(outSize, Vector(inSize)),
		bias(outSize)
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());

		std::normal_distribution<> dist(0.0, 0.1);

		for (auto& row : weights) {
			for (double& w : row) {
				w = dist(gen);
			}
		}

		for (double& b : bias) {
			b = dist(gen);
		}
	}

};


template<typename Activation>
class Model
{
private:
    std::vector<Parameters> m_parameters;
    std::vector<size_t> m_shape;

    Activation activation;   // stored functor

    Vector predict(const Vector& x, const Parameters& params) const;
    Matrix predictAll(const Matrix& x, const Parameters& params) const;
    Vector computeLayer(const Vector& input, size_t layerIndex) const;

public:
    double(*cost)(const Matrix&, const Matrix&);

    Model(std::initializer_list<size_t> shape,
        const Activation& activationFunc,
        double(*costFunc)(const Matrix&, const Matrix&))
        : activation(activationFunc), cost(costFunc)
    {
        if (shape.size() < 2)
            throw std::invalid_argument("Shape must contain at least input and output sizes.");

        m_shape = shape;

        for (size_t i = 0; i < shape.size() - 1; ++i)
            m_parameters.emplace_back(shape.begin()[i + 1], shape.begin()[i]);
    }

    Vector predict(const Vector& x) const;
    Matrix predictAll(const Matrix& x) const;

    std::vector<Parameters> gradient(double h, const Matrix& x, const Matrix& y);
    void train(const Matrix& x, const Matrix& y, double learningRate, size_t max_iterations);

    const std::vector<Parameters>& getParameters() const {
        return m_parameters;
    }
};

template<typename Activation>
Vector Model<Activation>::computeLayer(const Vector& input, size_t layerIndex) const
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

template<typename Activation>
Vector Model<Activation>::predict(const Vector& x) const
{
	Vector y = x;
	for (size_t i = 0; i < m_parameters.size(); ++i) {
		y = computeLayer(y, i);
		if (i < m_parameters.size() - 1) { // Don't apply ReLU to output layer
			y = activation(y);
		}
	}
	return y;
}

template<typename Activation>
Matrix Model<Activation>::predictAll(const Matrix& x) const
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

template<typename Activation>
std::vector<Parameters> Model<Activation>::gradient(double h, const Matrix& x, const Matrix& y)
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

template<typename Activation>
void Model<Activation>::train(const Matrix& x, const Matrix& y, double learningRate, size_t max_iterations)
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



double squareError(const Matrix& y_pred, const Matrix& y_true);

Vector ReLU(const Vector& input);
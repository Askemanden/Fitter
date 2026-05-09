#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <random>


typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;

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


class Model
{
private:
	std::vector<Parameters> m_parameters;
	std::vector<size_t> m_shape;

	Vector predict(const Vector& x, const Parameters& params) const;

	Matrix predictAll(const Matrix& x, const Parameters& params) const;

	Vector computeLayer(const Vector& input, size_t layerIndex) const;

	Vector(*activation)(const Vector&);

public:
	Model(std::initializer_list<size_t> shape, Vector(*activationFunc)(const Vector&)) {
		if (shape.size() < 2) {
			throw std::invalid_argument("Shape must contain at least input and output sizes.");
		}

		m_shape = shape;

		// offset one from the end due to shape containing output size 
		// as well as hidden layers
		for (size_t i = 0; i < shape.size() - 1; ++i) {
			Parameters params(shape.begin()[i + 1], shape.begin()[i]);
			m_parameters.push_back(params);
		}

		activation = activationFunc;
	}


	Vector predict(const Vector& x) const;

	Matrix predictAll(const Matrix& x) const;

	double cost(const Matrix& y_pred, const Matrix& y_true);

	std::vector<Parameters> gradient(double h, const Matrix& x, const Matrix& y);

	void train(const Matrix& x, const Matrix& y, double learningRate, size_t max_iterations);

	const std::vector<Parameters>& getParameters() const {
		return m_parameters;
	}
};

Vector ReLU(const Vector& input);
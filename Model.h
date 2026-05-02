#pragma once

#include <iostream>
#include <vector>

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;

typedef struct {
	Matrix weights;
	Vector bias;
} Parameters;

class Model
{
private:
	size_t m_inputSize;
	size_t m_outputSize;
	Parameters m_parameters;

	Vector predict(const Vector& x, const Parameters& params) const;

	Matrix predict(const Matrix& x, const Parameters& params) const;

public:
	Model(size_t inputSize, size_t outputSize) {
		m_inputSize = inputSize;
		m_outputSize = outputSize;
		m_parameters.weights = Matrix(outputSize, Vector(inputSize, 0));
		m_parameters.bias = Vector(outputSize, 0);
	}

	Vector predict(const Vector& x) const;

	Matrix predict(const Matrix& x) const;

	double cost(const Matrix& y_pred, const Matrix& y_true);

	Parameters gradient(double h, const Matrix& x, const Matrix& y);

	void train(const Matrix& x, const Matrix& y, double learningRate, size_t max_iterations);

	const Parameters& getParameters() const {
		return m_parameters;
	}
};


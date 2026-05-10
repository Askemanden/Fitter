#include "ActivationFunction.h"

Vector ActivationFunctions::ReLU::call(const Vector& layer) const
{
	Vector result = layer;
	for (double& v : result) {
		if (v < 0) {
			v = 0;
		}
	}
	return result;
}

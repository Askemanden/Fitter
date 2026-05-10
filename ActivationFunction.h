#pragma once

#include "Vector.h"

template<typename Child>
struct ActivationFunction
{
	inline Vector operator()(const Vector& layer) const {

		static_assert(std::is_invocable_r_v<Vector, const Child&, const Vector&>,
			"Child must implement: Vector call(const Vector&) const");

		return static_cast<const Child*>(this)->call(layer);
	};
};

namespace ActivationFunctions
{
	struct ReLU : public ActivationFunction<ReLU>
	{
		Vector call(const Vector& layer) const;
	};

	struct Sigmoid : public ActivationFunction<Sigmoid>
	{
		Vector call(const Vector& layer) const;
	};

	struct LeakyReLU : public ActivationFunction<LeakyReLU>
	{
		Vector call(const Vector& layer) const;
	};

	struct Tanh : public ActivationFunction<Tanh>
	{
		Vector call(const Vector& layer) const;
	};


}



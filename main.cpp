#include <iostream>
#include "autograd.hpp"

using autograd::backprop::Float32;
using autograd::backprop::Float64;

int main()
{
    const auto a = Float64{2};
    const auto b = Float64{3};

    const auto x = 5.0 * (4.0 + a) * b;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << x << std::endl;

    std::cout << "===================" << std::endl;

    x.Backward();
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << x << std::endl;
    return 0;
}

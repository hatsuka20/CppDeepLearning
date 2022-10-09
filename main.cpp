#include <iostream>
#include "autograd.hpp"

using autograd::backprop::Float32;

int main()
{
    auto a = Float32{2};
    auto b = Float32{3};
    auto c = Float32{4};
    auto d = Float32{5};

    // auto x = a * b + c * d;
    auto x = 2.0 * 3.0;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    std::cout << x << std::endl;

    std::cout << "===================" << std::endl;

    // x.Backward();
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    std::cout << x << std::endl;
    return 0;
}

#include <experimental/coroutine>
#include <iostream>

#include "tensor.hpp"

int main()
{
    std::size_t x = 5, y = 4, z = 3;
    Tensor<float> a{x, y, z};

    a(3U, 2U, 4U) = 5.0;
    std::cout << a(3U, 2U, 4U) << std::endl;

    return 0;
}

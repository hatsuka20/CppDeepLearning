#include <gtest/gtest.h>
#include "tensor.hpp"

TEST(TensorTest, InitializeTensor)
{
    Tensor x1{1, 2, 3};
    auto x2 = Tensor{1, 2, 3};
}
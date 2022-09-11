#pragma once
#include <vector>

template <class ElementType>
class Tensor
{
private:
    std::size_t dim_ = 0;
    std::vector<std::size_t> shape_ = {};
    ElementType* data_;

public:
    Tensor(std::initializer_list<ElementType> shape)
    {
        for (auto&& x : shape)
        {
            ++this->dim_;
            this->shape_.emplace_back(x);
        }
    }
};
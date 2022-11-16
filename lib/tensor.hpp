#pragma once
#include <iostream>
#include <numeric>
#include <vector>

template <class ElementType>
class Tensor
{
private:
    std::size_t dim_ = 0;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> shape_calc_;
    std::unique_ptr<ElementType[]> data_;

public:
    Tensor(std::initializer_list<std::size_t> shape)
    {
        for (auto&& x : shape)
        {
            ++dim_;
            shape_.emplace_back(x);
        }
        data_ = std::make_unique<ElementType[]>(std::reduce(shape_.cbegin(), shape_.cend(), std::size_t{0}));
        std::reduce(shape_.crbegin(), shape_.crend(), std::size_t{1}, [this](std::size_t acc, std::size_t x) {
            shape_calc_.emplace_back(acc);
            return acc * x;
        });
        std::reverse(shape_calc_.begin(), shape_calc_.end());
    }

    template <class... Args>
    auto operator()(Args... args) -> decltype(std::initializer_list<std::size_t>{args...}, std::declval<ElementType&>())
    {
        auto indexes = std::vector<std::size_t>{args...};
        const auto true_index =
            std::inner_product(indexes.cbegin(), indexes.cend(), shape_calc_.cbegin(), std::size_t{0});
        return data_[true_index];
    }
};

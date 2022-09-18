#include <iostream>
#include <string>
#include "autograd.hpp"
#include "tensor.hpp"

class Layer
{
private:
    const std::string name = "Layer";

public:
    Layer() = default;
    auto GetLayerName() { return this->name; }
    virtual void Forward() = 0;
};

class DenseLayer : public Layer
{
private:
    const std::string name = "DenseLayer";

public:
    DenseLayer() = default;
    void Forward() override { std::cout << "DenseLayer: Forward" << std::endl; }
};

int main()
{
    auto tensor = Tensor<float>{3, 2, 4};
    auto layer = DenseLayer();
    layer.Forward();

    auto a = autograd::Float32{1.0};
    auto b = autograd::Float32{2.0};
    auto c = autograd::Float32{3.0};

    auto x = a + b + c + a + b + c;

    std::cout << x << std::endl;
    return 0;
}
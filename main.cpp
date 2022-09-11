#include <iostream>
#include <string>
#include <vector>
#include "tensor.hpp"

class Layer
{
private:
    const std::string name = "Layer";

public:
    Layer() = default;
    auto GetLayerName() { return this->name; }
    virtual void Forward() = 0;
    virtual void Backward() = 0;
};

class DenseLayer : public Layer
{
private:
    const std::string name = "DenseLayer";

public:
    DenseLayer() = default;
    void Forward() override { std::cout << "DenseLayer: Forward" << std::endl; }
    void Backward() override { std::cout << "DenseLayer: Backward" << std::endl; }
};

int main()
{
    auto tensor = Tensor<float>{3, 2, 4};
    auto layer = DenseLayer();
    layer.Forward();
    layer.Backward();

    return 0;
}
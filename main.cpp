#include <iostream>
#include <string>

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
    auto layer = DenseLayer();
    layer.Forward();
    layer.Backward();

    return 0;
}
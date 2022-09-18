#include <iostream>
#include "nameof.hpp"

namespace autograd
{
    template <class T>
    class Operand
    {
    public:
        [[nodiscard]] virtual T Forward() const = 0;
        virtual void Backward(const T&) const = 0;
    };

    template <class T>
    class Terminal : public Operand<T>
    {
        [[nodiscard]] T Forward() const override
        {
            std::cout << "Terminal consturctor" << std::endl;
            return T{0};
        }
        void Backward(const T&) const override {}
    };

    template <class T>
    class Add : public Operand<T>
    {
    private:
        const T& term1_;
        const T& term2_;

    public:
        explicit Add(const T& src1, const T& src2) : term1_(src1), term2_(src2)
        {
            std::cout << "Add consturctor" << std::endl;
        }

        [[nodiscard]] T Forward() const override { return T{static_cast<float>(term1_) + static_cast<float>(term2_)}; }

        void Backward(const T& dL) const override {}
    };

    class Float32
    {
    private:
        const Operand<Float32>& backward_type_;
        const float x;

    public:
        explicit Float32(float src) : backward_type_(std::move(Terminal<Float32>())), x(src)
        {
            std::cout << "Normal Float32 consturctor" << std::endl;
        }
        Float32(const Float32& src) : backward_type_(std::move(Terminal<Float32>())), x(src.x)
        {
            std::cout << "const ref Float32 consturctor" << std::endl;
        }
        explicit Float32(const Operand<Float32>& op) : backward_type_(op), x(backward_type_.Forward().x)
        {
            std::cout << "const ref Operand consturctor" << std::endl;
        }
        ~Float32() = default;

        explicit operator float() const noexcept { return x; }

        friend std::ostream& operator<<(std::ostream& ofs, const Float32& src)
        {
            ofs << src.x << " (Float32, backward=" << NAMEOF_SHORT_TYPE_RTTI(src.backward_type_) << ")";
            return ofs;
        }
    };

    template <class T>
    auto operator+(const T& a, const T& b)
    {
        return T{Add(a, b)};
    }
};  // namespace autograd
#pragma once

#include <iostream>
#include <memory>
#include "debug.hpp"
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
    class Add : public Operand<T>
    {
    private:
        const T& term1_;
        const T& term2_;

    public:
        explicit Add(const T& src1, const T& src2) : term1_(src1), term2_(src2) { debug_print("Add consturctor"); }

        [[nodiscard]] T Forward() const override
        {
            return T{static_cast<typename T::RawValueType>(term1_) + static_cast<typename T::RawValueType>(term2_)};
        }

        void Backward(const T& d_l) const override {}
    };

    template <class T>
    class Mul : public Operand<T>
    {
    private:
        const T& term1_;
        const T& term2_;

    public:
        explicit Mul(const T& src1, const T& src2) : term1_(src1), term2_(src2) { debug_print("Mul consturctor"); }

        [[nodiscard]] T Forward() const override
        {
            return T{static_cast<typename T::RawValueType>(term1_) * static_cast<typename T::RawValueType>(term2_)};
        }

        void Backward(const T& d_l) const override {}
    };

    class Float32
    {
    public:
        using RawValueType = float;
        explicit operator RawValueType() const noexcept { return x_; }

    private:
        std::shared_ptr<Operand<Float32>> backward_type_;
        RawValueType x_;

    public:
        explicit Float32(float src) : backward_type_(nullptr), x_(src)
        {
            debug_print("Float32(float src) consturctor");
        }

        Float32(const Float32& src) : backward_type_(src.backward_type_), x_(src.x_)
        {
            debug_print("Float32(const Float32& src) consturctor");
        }

        explicit Float32(const std::shared_ptr<Operand<Float32>>& op)
            : backward_type_(op), x_(backward_type_->Forward().x_)
        {
            debug_print("Float32(Operand<Float32>* op) consturctor");
        }

        ~Float32() { debug_print("Float32 destructor"); };

        friend auto operator<<(std::ostream& ofs, const Float32& src) -> decltype(ofs)
        {
            auto& temp = *src.backward_type_;
            ofs << src.x_
                << " (Float32, backward=" << ((src.backward_type_ == nullptr) ? "None" : NAMEOF_SHORT_TYPE_RTTI(temp))
                << ")";
            return ofs;
        }

        [[nodiscard]] auto Detach() const { return Float32{x_}; }
        [[nodiscard]] auto Clone() const { return Float32{*this}; }
    };

    template <class T>
    auto operator+(const T& a, const T& b)
    {
        return T{std::make_shared<Add<T>>(a, b)};
    }

    template <class T>
    auto operator*(const T& a, const T& b)
    {
        return T{std::make_shared<Mul<T>>(a, b)};
    }
};  // namespace autograd
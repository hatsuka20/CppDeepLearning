#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "debug.hpp"

namespace autograd
{
    namespace backprop
    {
        class Float32
        {
        public:
            using ElementType = float;
            using GradType = float;
            explicit operator ElementType() const { return x_; }

        private:
            ElementType x_;
            mutable std::optional<GradType> d_;
            mutable std::string last_calculate_ = "Leaf";
            mutable std::vector<std::pair<std::reference_wrapper<const Float32>, GradType>> children_;

            void Backward(const GradType& dl) const
            {
                d_ = d_ ? d_.value() + dl : dl;
                for (const auto& c : children_)
                {
                    c.first.get().Backward(d_.value() * c.second);
                }
            }

        public:
            Float32() = delete;
            explicit Float32(const ElementType& x) : x_(x) {}
            Float32(const ElementType& x, const std::string& last_calculate) : x_(x), last_calculate_(last_calculate) {}
            Float32(const Float32& src) = default;
            Float32(Float32&& src) noexcept
                : x_(src.x_),
                  d_(src.d_),
                  last_calculate_(std::move(src.last_calculate_)),
                  children_(std::move(src.children_))
            {
            }
            ~Float32() = default;

            void AddChild(const Float32& src, const GradType& d) const { children_.emplace_back(src, d); }
            auto& GetChildren() const { return children_; }

            void Backward() const
            {
                for (const auto& c : children_)
                {
                    c.first.get().Backward(c.second);
                }
            }

            friend auto& operator<<(std::ostream& ofs, const Float32& src)
            {
                if (src.d_)
                {
                    ofs << src.x_ << " (grad=" << src.d_.value() << ", backward=" << src.last_calculate_ << ")";
                }
                else
                {
                    ofs << src.x_ << " (grad=None, backward=" << src.last_calculate_ << ")";
                }
                return ofs;
            }
        };

        template <class T, class U>
        decltype(auto) Mul(T&& lhs, U&& rhs)
        {
            const auto lhsx = typename std::remove_reference_t<T>::ElementType(lhs);
            const auto rhsx = typename std::remove_reference_t<T>::ElementType(rhs);

            auto t = std::remove_reference_t<T>{lhsx * rhsx, __func__};
            if constexpr (std::is_lvalue_reference_v<decltype(lhs)>)
            {
                t.AddChild(lhs, rhsx);
            }
            else
            {
                for (const auto& c : lhs.GetChildren())
                {
                    t.AddChild(c.first, c.second * rhsx);
                }
            }

            if constexpr (std::is_lvalue_reference_v<decltype(rhs)>)
            {
                t.AddChild(rhs, lhsx);
            }
            else
            {
                for (const auto& c : rhs.GetChildren())
                {
                    t.AddChild(c.first, c.second * lhsx);
                }
            }

            return t;
        }

        template <class T, class U>
        decltype(auto) Add(T&& lhs, U&& rhs)
        {
            const auto lhsx = typename std::remove_reference_t<T>::ElementType(lhs);
            const auto rhsx = typename std::remove_reference_t<T>::ElementType(rhs);

            auto t = std::remove_reference_t<T>{lhsx + rhsx, __func__};
            if constexpr (std::is_lvalue_reference_v<decltype(lhs)>)
            {
                t.AddChild(lhs, 1);
            }
            else
            {
                for (const auto& c : lhs.GetChildren())
                {
                    t.AddChild(c.first, c.second);
                }
            }

            if constexpr (std::is_lvalue_reference_v<decltype(rhs)>)
            {
                t.AddChild(rhs, 1);
            }
            else
            {
                for (const auto& c : rhs.GetChildren())
                {
                    t.AddChild(c.first, c.second);
                }
            }

            return t;
        }

        template <class T, class U>
        auto operator+(T&& lhs, U&& rhs)
        {
            return Add(std::forward<T>(lhs), std::forward<U>(rhs));
        }

        template <class T, class U>
        auto operator*(T&& lhs, U&& rhs)
        {
            return Mul(std::forward<T>(lhs), std::forward<U>(rhs));
        }
    }  // namespace backprop
};     // namespace autograd
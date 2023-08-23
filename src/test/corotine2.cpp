#include <coroutine>
#include <iostream>
#include <vector>
#include <cstdio>
#include "base/timer.h"

template <typename T = void>
struct [[nodiscard]] generator
{
    struct promise_type;
    using handle = std::coroutine_handle<promise_type>;

    struct promise_type
    {
        promise_type() {}
        ~promise_type()
        {
            reinterpret_cast<T*>(&ret_val_buf_)->~T();
        }
        auto get_return_object() { return generator{ handle::from_promise(*this) }; }
        auto initial_suspend() { return std::suspend_never{}; }
        auto final_suspend() noexcept { return std::suspend_always{}; }
        void unhandled_exception() { std::terminate(); }
        void return_value(const T value)
        {
            new (&ret_val_buf_) T(std::move(value));
        }
        T&& transfer_return_value()
        {
            return std::move(*reinterpret_cast<T*>(&ret_val_buf_));
        }
        T get_return_value()
        {
            return *reinterpret_cast<T*>(&ret_val_buf_);
        }
        // void *operator new(size_t sz) { return coroutine_allocator.alloc(sz); }
        // void operator delete(void *p, size_t sz) { coroutine_allocator.free(p, sz); }
        struct alignas(alignof(T)) T_Buf
        {
            uint8_t buf[sizeof(T)];
        };

        std::coroutine_handle<> callee_coro = nullptr;
        T_Buf ret_val_buf_;
    };

    auto get_handle()
    {
        auto result = coro;
        coro = nullptr;
        return result;
    }

    generator(generator const&) = delete;
    generator(handle h = nullptr): coro(h) {}
    generator(generator&& rhs): coro(rhs.coro) { rhs.coro = nullptr; }
    ~generator()
    {
        if (coro)
        {
            coro.destroy();
        }
    }

    generator& operator=(generator const&) = delete;
    generator& operator=(generator&& rhs)
    {
        if (this != &rhs)
        {
            coro = rhs.coro;
            rhs.coro = nullptr;
        }
        return *this;
    }

    struct awaiter
    {
        awaiter(handle h): awaiter_coro(h) {}
        constexpr bool await_ready() const noexcept { return false; }
        template <typename awaiting_handle>
        constexpr void await_suspend(awaiting_handle awaiting_coro) noexcept
        {
            awaiting_coro.promise().callee_coro = awaiter_coro;
        }
        constexpr auto await_resume() noexcept
        {
            return awaiter_coro.promise().transfer_return_value();
        }

    private:
        handle awaiter_coro;
    };

    auto operator co_await() { return awaiter(coro); }

private:
    handle coro;
};

const int Maxcorotines = 8;
const int test_count = 10000;
int global_in[Maxcorotines];

inline generator<bool> display_coro(int& para, int coro_id)
{
    int cnt = 0;
    int step = para;
    while (true)
    {
        // std::cout << para << ' ' << (cnt+=step) << std::endl;
        para = (cnt+=step);
        co_await std::suspend_always{};
        global_in[coro_id] = para;
    }
    co_return true;
}


inline generator<bool> test(int& para, int coro_id) {
    int cnt = 0;
    int step = para;
    while (true)
    {
        // std::cout << para << ' ' << (cnt+=step) << std::endl;
        para = (cnt+=step);
        co_await std::suspend_always{};
        global_in[coro_id] = para;
    }
    co_return true;
}

void test2(int& para, int coro_id)
{
    auto x = test(para, coro_id);
    return;
}

inline bool run_display_coro(int para)
{
    std::vector<std::coroutine_handle<>> handles;
    int i_in_master[Maxcorotines];
    for (int i = 0; i < Maxcorotines; ++i)
    {
        i_in_master[i] = i;
        handles.emplace_back(display_coro(i_in_master[i], i).get_handle());
    }
    puts("begin running");
    xmh::Timer timer("t1");
    
    timer.begin();
    for (int i = 0; i < test_count * Maxcorotines; ++i)
    {
        handles[i % Maxcorotines].resume();
    }
    timer.end();
    double res = timer.ManualQuery("t1");

    puts("----------result show----------");
    for (int i = 0; i < Maxcorotines; i++)
    {
        std::cout << i_in_master[i] << " " << global_in[i] << std::endl;
    }
    puts("----------perf----------");
    std::cout << res * 1.0 / (test_count * Maxcorotines) << "ns per [2-switch]" << std::endl;
    return true;
}

int main()
{
    bool ret = false;
    ret = run_display_coro(0);
    return 0;
}
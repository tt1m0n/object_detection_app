#ifndef FRAMEQUEUE_HPP_
#define FRAMEQUEUE_HPP_

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class FrameQueue : public std::queue<T>
{
public:
    FrameQueue() : counter_(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::queue<T>::push(entry);
        counter_++;
        cond_var.notify_one(); // Notify a waiting consumer
    }

    T get()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_var.wait(lock, [this]() { return this->counter_ > 0; }); // Wait until there's data

        T entry = this->front();
        this->pop();
        counter_--;
        return entry;
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return this->counter_;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

private:
    unsigned int counter_;
    std::mutex mutex;
    std::condition_variable cond_var;
};


#endif // FRAMEQUEUE_HPP_
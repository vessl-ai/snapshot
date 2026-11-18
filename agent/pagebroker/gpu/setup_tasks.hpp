// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <system_error>
#include <thread>
#include <vector>

namespace snapshot::pagebroker::gpu {
// Run task(index) once for every index in [0, count) and return when all have
// finished. `threads` bounds how many run at once: 1 runs them in index order on
// the calling thread and stops at the first exception, as a plain loop does;
// 0 gives every index its own thread. With more than one thread, a task that
// throws stops the others from starting new indices, every started task still
// runs to completion, and the exception of the lowest index that threw is then
// rethrown. If a thread cannot be created, the tasks run on fewer threads.
inline void
RunSetupTasks(size_t count, size_t threads, const std::function<void(size_t)>& task)
{
  if (threads == 0 || threads > count) {
    threads = count;
  }
  if (threads <= 1) {
    for (size_t index = 0; index < count; ++index) {
      task(index);
    }
    return;
  }
  std::vector<std::exception_ptr> failures(count);
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  const auto work = [&] {
    while (!failed.load(std::memory_order_acquire)) {
      const size_t index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) {
        return;
      }
      try {
        task(index);
      } catch (...) {
        // Each index has its own slot, and joining the workers publishes them.
        failures[index] = std::current_exception();
        failed.store(true, std::memory_order_release);
      }
    }
  };
  {
    // The calling thread is one of the workers.
    std::vector<std::jthread> workers;
    workers.reserve(threads - 1);
    for (size_t worker = 1; worker < threads; ++worker) {
      try {
        workers.emplace_back(work);
      } catch (const std::system_error&) {
        break;
      }
    }
    work();
  }
  for (const auto& failure : failures) {
    if (failure) {
      std::rethrow_exception(failure);
    }
  }
}
}  // namespace snapshot::pagebroker::gpu

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "setup_tasks.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace snapshot::pagebroker::gpu {
namespace {
using namespace std::chrono_literals;

// Counts tasks that have started and lets each one wait, with a bound, until a
// given number have started, which is only possible if they run at once.
class Rendezvous {
 public:
  explicit Rendezvous(size_t expected) : expected_(expected) {}

  bool ArriveAndWait()
  {
    std::unique_lock lock(mutex_);
    ++arrived_;
    changed_.notify_all();
    return changed_.wait_for(lock, 10s, [&] { return arrived_ >= expected_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  size_t expected_;
  size_t arrived_ = 0;
};

TEST(RunSetupTasksTest, SerialRunsInOrderOnCallingThreadAndStopsAtFirstFailure)
{
  std::vector<size_t> order;
  const auto caller = std::this_thread::get_id();
  bool same_thread = true;
  try {
    RunSetupTasks(5, 1, [&](size_t index) {
      same_thread = same_thread && std::this_thread::get_id() == caller;
      order.push_back(index);
      if (index == 2) {
        throw std::logic_error("task 2");
      }
    });
    FAIL() << "the failure was not rethrown";
  } catch (const std::logic_error& error) {
    EXPECT_STREQ(error.what(), "task 2");
  }
  EXPECT_TRUE(same_thread);
  EXPECT_EQ(order, (std::vector<size_t>{0, 1, 2}));
}

TEST(RunSetupTasksTest, ZeroThreadsRunsEveryTaskAtOnce)
{
  constexpr size_t kTasks = 8;
  Rendezvous rendezvous(kTasks);
  std::vector<int> runs(kTasks, 0);
  std::atomic<size_t> met{0};
  RunSetupTasks(kTasks, 0, [&](size_t index) {
    ++runs[index];
    if (rendezvous.ArriveAndWait()) {
      ++met;
    }
  });
  EXPECT_EQ(met.load(), kTasks) << "tasks did not all run at the same time";
  EXPECT_EQ(runs, std::vector<int>(kTasks, 1));
}

TEST(RunSetupTasksTest, BoundedThreadsRunEveryTaskOnceWithinTheBound)
{
  constexpr size_t kTasks = 9;
  constexpr size_t kThreads = 3;
  std::vector<int> runs(kTasks, 0);
  std::atomic<size_t> active{0};
  std::atomic<size_t> peak{0};
  RunSetupTasks(kTasks, kThreads, [&](size_t index) {
    const size_t now = ++active;
    size_t seen = peak.load();
    while (now > seen && !peak.compare_exchange_weak(seen, now)) {
    }
    ++runs[index];
    std::this_thread::sleep_for(5ms);
    --active;
  });
  EXPECT_LE(peak.load(), kThreads);
  EXPECT_EQ(runs, std::vector<int>(kTasks, 1));
}

TEST(RunSetupTasksTest, MoreThreadsThanTasksAndNoTasks)
{
  std::vector<int> runs(2, 0);
  RunSetupTasks(2, 16, [&](size_t index) { ++runs[index]; });
  EXPECT_EQ(runs, std::vector<int>(2, 1));
  bool called = false;
  RunSetupTasks(0, 0, [&](size_t) { called = true; });
  RunSetupTasks(0, 1, [&](size_t) { called = true; });
  EXPECT_FALSE(called);
}

TEST(RunSetupTasksTest, ParallelFailureWaitsForStartedTasksAndRethrowsLowestIndex)
{
  constexpr size_t kTasks = 4;
  Rendezvous rendezvous(kTasks);
  std::vector<int> finished(kTasks, 0);
  try {
    RunSetupTasks(kTasks, 0, [&](size_t index) {
      // Every task starts before any of them throws.
      rendezvous.ArriveAndWait();
      if (index == 3) {
        throw std::runtime_error("task 3");
      }
      if (index == 1) {
        std::this_thread::sleep_for(20ms);
        throw std::logic_error("task 1");
      }
      std::this_thread::sleep_for(20ms);
      finished[index] = 1;
    });
    FAIL() << "the failure was not rethrown";
  } catch (const std::logic_error& error) {
    EXPECT_STREQ(error.what(), "task 1");
  }
  EXPECT_EQ(finished, (std::vector<int>{1, 0, 1, 0}));
}

TEST(RunSetupTasksTest, ParallelFailureStopsUnstartedTasks)
{
  constexpr size_t kTasks = 6;
  std::atomic<size_t> started{0};
  try {
    RunSetupTasks(kTasks, 2, [&](size_t index) {
      ++started;
      if (index == 0) {
        throw std::runtime_error("task 0");
      }
      std::this_thread::sleep_for(50ms);
    });
    FAIL() << "the failure was not rethrown";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "task 0");
  }
  // Task 0 throws before its worker takes another index, and the other worker
  // checks for the failure before every index it takes.
  EXPECT_LE(started.load(), 3u);
}
}  // namespace
}  // namespace snapshot::pagebroker::gpu

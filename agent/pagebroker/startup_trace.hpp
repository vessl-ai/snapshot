// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace snapshot::pagebroker {
// PAGEBROKER_STARTUP_TRACE set to anything but "" or "0" prints one stderr line
// per startup step, so the time before the socket opens can be split into
// driver initialization, context creation and ring pinning:
//   pagebroker-startup <unix seconds> +<seconds since the first line> <step>
// The wall-clock column lines the steps up with a supervisor's own log.
inline bool
StartupTraceEnabled()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PAGEBROKER_STARTUP_TRACE");
    return value && *value && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

[[gnu::format(printf, 1, 2)]] inline void
StartupTrace(const char* format, ...)
{
  if (!StartupTraceEnabled()) {
    return;
  }
  timespec wall{};
  timespec now{};
  clock_gettime(CLOCK_REALTIME, &wall);
  clock_gettime(CLOCK_MONOTONIC, &now);
  static const timespec origin = now;
  const double since = static_cast<double>(now.tv_sec - origin.tv_sec) +
      static_cast<double>(now.tv_nsec - origin.tv_nsec) / 1e9;
  char line[512];
  int length = std::snprintf(line, sizeof(line), "pagebroker-startup %lld.%06ld +%.6f ",
                             static_cast<long long>(wall.tv_sec), wall.tv_nsec / 1000, since);
  if (length < 0 || length >= static_cast<int>(sizeof(line)) - 1) {
    return;
  }
  va_list arguments;
  va_start(arguments, format);
  const int message = std::vsnprintf(line + length, sizeof(line) - static_cast<size_t>(length), format, arguments);
  va_end(arguments);
  if (message > 0) {
    length += message;
  }
  if (length > static_cast<int>(sizeof(line)) - 2) {
    length = static_cast<int>(sizeof(line)) - 2;
  }
  line[length++] = '\n';
  // One write per line keeps lines from concurrent setup threads whole.
  const ssize_t written = write(STDERR_FILENO, line, static_cast<size_t>(length));
  (void)written;
}
}  // namespace snapshot::pagebroker

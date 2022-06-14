#pragma once
#include <iomanip>
#include <iostream>

#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <g3log/loglevels.hpp>

#include "customSink.hh"

class Logger {
  public:
    Logger();
    ~Logger();

    void destroy();

  private:
    std::unique_ptr<g3::LogWorker> logWorker;
    std::unique_ptr<CustomSink> customSink;
    void initialize();
};

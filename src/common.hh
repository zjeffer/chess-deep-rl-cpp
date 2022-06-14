#pragma once
#undef LOG
#include "logger/logger.hh"
#include <random>

inline bool g_running = true;

inline std::default_random_engine g_generator;
inline std::gamma_distribution<double> g_distribution(0.3, 1.0);

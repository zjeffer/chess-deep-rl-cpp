#pragma once
#undef LOG
#include "logger/logger.hh"
#include <random>

inline bool g_running = true;
inline bool g_isSelfPlaying = false;

inline std::unique_ptr<Logger> logger = std::make_unique<Logger>();

inline std::default_random_engine g_generator;
inline std::gamma_distribution<double> g_distribution(0.3, 1.0);

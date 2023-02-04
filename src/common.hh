#pragma once
#undef LOG
#include "logger/logger.hh"
#include <random>

inline bool g_Running = true;
inline bool g_IsSelfPlaying = false;

inline std::shared_ptr<Logger> g_Logger;

inline std::default_random_engine g_Generator;
inline std::gamma_distribution<double> g_distribution(0.3, 1.0);

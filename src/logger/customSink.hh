#pragma once

#include <string>
#include <iostream>
#include <g3log/logmessage.hpp>
#include <iomanip>


enum LogColor {
	RESET,
	WHITE = 97,
	BLUE = 94,
	GREEN = 32, 
	YELLOW = 33, 
	RED = 31
};

#define ADD_COLOR(color) "\x1b[" << int(color) << "m"
#define RESET_COLOR "\x1b[" << int(LogColor::RESET) << "m"

struct CustomSink {
	void callback(g3::LogMessageMover log){
		std::cout << log.get().toString(CustomSink::FormatMsg) << std::flush;
	}

	static LogColor getColor(const LEVELS level){
		switch(level.value){
			case g3::kDebugValue:
				return LogColor::BLUE;
			case g3::kInfoValue:
				return LogColor::GREEN;
			case g3::kWarningValue:
				return LogColor::YELLOW;
			case g3::kFatalValue:
				return LogColor::RED;
		}
		return g3::internal::wasFatal(level) ? LogColor::RED : LogColor::WHITE;
	}

	static std::string FormatMsg(const g3::LogMessage& msg) {
		std::stringstream ss;
		LogColor color = CustomSink::getColor(msg._level);
		ss << ADD_COLOR(color) 
			<< "[" << msg.timestamp("%H:%M:%S") << "]"
			<< "[" << std::setw(7) << msg.level() << "/"
			<< msg.file() << ":" << msg.line() << "] "
			<< RESET_COLOR;
		return ss.str();
	}
};
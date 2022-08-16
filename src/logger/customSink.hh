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

class CustomSink {
  public:
	virtual ~CustomSink() = default;

	virtual void callback(g3::LogMessageMover log) = 0;

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
			default: 
				return g3::internal::wasFatal(level) ? LogColor::RED : LogColor::WHITE;
		}
	} 
};

class StdoutSink : public CustomSink {
  public:
	void callback(g3::LogMessageMover log) override {
		std::cout << log.get().toString(&FormatMsg) << std::flush;
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

class MainWindow;

class QtConsoleSink : public CustomSink {
  public:
    QtConsoleSink(MainWindow *window);

    void callback(g3::LogMessageMover log) override;

	static std::string FormatMsg(const g3::LogMessage& msg);

  private:
    MainWindow *m_Window;
};
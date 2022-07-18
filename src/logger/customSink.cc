#include "customSink.hh"
#include "../ui/mainwindow.hh"

QtConsoleSink::QtConsoleSink(MainWindow* window) {
	m_Window = window;
}

void QtConsoleSink::callback(g3::LogMessageMover log) {
	m_Window->print(log.get().toString(&FormatMsg));
}

std::string QtConsoleSink::FormatMsg(const g3::LogMessage& msg) {
	std::stringstream ss;
	ss << "[" << msg.timestamp("%H:%M:%S") << "]"
		<< "[" << std::setw(7) << msg.level() << "/"
		<< msg.file() << ":" << msg.line() << "] ";
	return ss.str();
}
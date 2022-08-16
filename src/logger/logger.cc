#include "logger.hh"
#include "customSink.hh"

Logger::Logger()
	: logWorker(g3::LogWorker::createLogWorker()),
	  stdoutSink(std::make_unique<StdoutSink>()) {
	logWorker->addSink(std::move(stdoutSink), &StdoutSink::callback);

	g3::initializeLogging(logWorker.get());
}

Logger::Logger(MainWindow *mainWindow)
	: logWorker(g3::LogWorker::createLogWorker()),
	  stdoutSink(std::make_unique<StdoutSink>()),
	  qtConsoleSink(std::make_unique<QtConsoleSink>(mainWindow)) {

	logWorker->addSink(std::move(stdoutSink), &StdoutSink::callback);
	logWorker->addSink(std::move(qtConsoleSink), &QtConsoleSink::callback);

	g3::initializeLogging(logWorker.get());
}

Logger::~Logger() { 
	// this->destroy();
}

void Logger::destroy() {
	std::cout << "Destroying logger" << std::endl;
	logWorker->removeAllSinks();
	logWorker.reset();
}
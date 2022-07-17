#include "logger.hh"



Logger::Logger(MainWindow* mainWindow) {
	logWorker = g3::LogWorker::createLogWorker();
	stdoutSink = std::make_unique<StdoutSink>();
	qtConsoleSink = std::make_unique<QtConsoleSink>(mainWindow);
	logWorker->addSink(std::move(stdoutSink), &StdoutSink::callback);

	g3::initializeLogging(logWorker.get());
}

Logger::~Logger(){
	this->destroy();
}

void Logger::destroy(){
	std::cout << "Destroying logger" << std::endl;
	logWorker->removeAllSinks();
	logWorker.reset();
}
#include "logger.hh"



Logger::Logger() {
	initialize();
}

Logger::~Logger(){
}

void Logger::destroy(){
	std::cout << "Destroying logger" << std::endl;
	logWorker->removeAllSinks();
	logWorker.reset();
}

void Logger::initialize() {
	logWorker = g3::LogWorker::createLogWorker();
	customSink = std::make_unique<CustomSink>();
	logWorker->addSink(std::move(customSink), &CustomSink::callback);

	g3::initializeLogging(logWorker.get());
}
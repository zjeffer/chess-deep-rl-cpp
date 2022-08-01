#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

#include <iostream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "train.hh"
#include "logger/logger.hh"

#include "ui/mainwindow.hh"
#include <QApplication>
#include <thread>

#include "common.hh"

std::unique_ptr<MainWindow> g_mainWindow;

void signal_handling(int signal) {
	LOG(INFO) << "Signal " << signal << " received. Quitting...";
	g_running = false;

	if (g_mainWindow != nullptr) {
		g_mainWindow->close();
		g_mainWindow.reset();
	}
}

// argument parsing (https://stackoverflow.com/a/868894/10180569)
class InputParser {
  public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            this->tokens.push_back(std::string(argv[i]));
        }
    }

    /// @author iain
    const std::string &getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    /// @author iain
    bool cmdOptionExists(const std::string &option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

  private:
    std::vector<std::string> tokens;
};

void printUsage(char* filename) {
	// print help and exit
	std::cout << "Usage: " << filename << " [options]" << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "  -h, --help\t\t\tPrint this help message" << std::endl;
	std::cout << "  --console\t\t\tRun in console mode" << std::endl;
	std::cout << "  --sims\t\t\tAmount of simulations" << std::endl;
	std::cout << "  --parallel-games\t\tAmount of games to play in parallel" << std::endl;
	std::cout << "  --train <model_path>\t\tTrain the model" << std::endl;
	exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {
	// signal handling
	signal(SIGINT, signal_handling);
	signal(SIGTERM, signal_handling);

	// defaults
	int amount_of_sims = 50;
	int parallel_games = 1;
	int return_code = 0;

	// parse arguments
	InputParser inputParser = InputParser(argc, argv);
	if (inputParser.cmdOptionExists("-h") || inputParser.cmdOptionExists("--help")) {
		printUsage(argv[0]);
	}

	if (inputParser.cmdOptionExists("--train")) {
		// get network path
		logger = std::make_shared<Logger>();
		const std::string& network_path = inputParser.getCmdOption("--train");
		Trainer trainer(network_path);
		trainer.train();
	} else if (inputParser.cmdOptionExists("--console")) {
		// sims
		if (inputParser.getCmdOption("--sims") != "") {
			try {
				amount_of_sims = std::stoi(inputParser.getCmdOption("--sims"));
				if (amount_of_sims < 1) {
					throw std::invalid_argument("Amount must be greater or equal to 1");
				}
			} catch (std::invalid_argument& e) {
				std::cerr << "Invalid amount of simulations. Make sure you entered a number >= 1" << std::endl;
				std::cerr << e.what() << std::endl;
				exit(EXIT_FAILURE);
			}
		} else {
			std::cout << "No amount of simulations specified. Using default value of " << amount_of_sims << "." << std::endl;
		}

		// parallel games
		if (inputParser.getCmdOption("--parallel-games") != "") {
			try {
				parallel_games = std::stoi(inputParser.getCmdOption("--parallel-games"));
				if (parallel_games < 1) {
					throw std::invalid_argument("Amount must be greater or equal to 1");
				}
			} catch (std::invalid_argument& e) {
				std::cerr << "Invalid amount of parallel games. Make sure you entered a number >= 1" << std::endl;
				std::cerr << e.what() << std::endl;
				exit(EXIT_FAILURE);
			}
		} else {
			std::cout << "No amount of parallel games specified. Using default value of " << parallel_games << "." << std::endl;
		}

		
		logger = std::make_shared<Logger>();
		
		// TODO: write code to run selfplay without GUI
		LOG(DEBUG) << "Running in console mode!";
	} else {
		// GUI app
		QApplication a(argc, argv);
		g_mainWindow = std::make_unique<MainWindow>();
		g_mainWindow->show();
		return_code = a.exec();
	}
	

	return return_code;
}
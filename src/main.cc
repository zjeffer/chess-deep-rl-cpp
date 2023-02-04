#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

#include <iostream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "node.hh"
#include "selfplay.hh"
#include "train.hh"
#include "logger/logger.hh"

#include "ui/mainwindow.hh"
#include <QApplication>
#include <thread>

#include "common.hh"

std::unique_ptr<MainWindow> g_mainWindow = nullptr;

void signal_handling(int signal) {
	std::cerr << "Signal " << signal << " received. Quitting..." << std::endl;
	g_IsSelfPlaying = false;
	g_Running = false;
}

// argument parsing (https://stackoverflow.com/a/868894/10180569)
class InputParser {
  public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            m_Tokens.push_back(std::string(argv[i]));
        }
    }

    /// @author iain
    const std::string &getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(m_Tokens.begin(), m_Tokens.end(), option);
        if (itr != m_Tokens.end() && ++itr != m_Tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    /// @author iain
    bool cmdOptionExists(const std::string &option) const {
        return std::find(m_Tokens.begin(), m_Tokens.end(), option) != m_Tokens.end();
    }

  private:
    std::vector<std::string> m_Tokens;
};

void printUsage(char* filename) {
	// print help and exit
	std::cout << "Usage: " << filename << " [options]" << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "  -h, --help\t\t\tPrint this help message" << std::endl;
	std::cout << "  --console\t\t\tRun in console mode" << std::endl;
	std::cout << "  --sims\t\t\tAmount of simulations" << std::endl;
	std::cout << "  --parallel-games\t\tAmount of games to play in parallel" << std::endl;
	std::cout << "  --train\t\tTrain the model. Needs a --model parameter" << std::endl;
	std::cout << "  --model\t\tModel to train/run selfplay with" << std::endl;
	std::cout << "  --bs\t\t\tBatch size (for training)" << std::endl;
	std::cout << "  --lr\t\t\tLearning rate (for training)" << std::endl;
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
		exit(EXIT_SUCCESS);
	}

	if (inputParser.cmdOptionExists("--test")) {
		g_Logger = std::make_shared<Logger>();
		// TODO: create unit tests
		exit(EXIT_SUCCESS);
	}

	if (inputParser.cmdOptionExists("--train")) {
		// get network path
		std::string networkPath = "";
		if (inputParser.getCmdOption("--model") != "") {
			try {
				networkPath = inputParser.getCmdOption("--model");				
			} catch (const std::exception& e) {
				std::cerr << "Could not load model. Make sure you entered a valid path" << std::endl;
				std::cerr << e.what() << std::endl;
				exit(EXIT_FAILURE);
			}
		} else {
			std::cerr << "No model path specified" << std::endl;
			exit(EXIT_FAILURE);
		}

		int batchSize = 128;
		float learningRate = 0.02;

		if (inputParser.cmdOptionExists("--bs")) {
			try {
				batchSize = std::stoi(inputParser.getCmdOption("--bs"));
			} catch (const std::exception& e) {
				std::cerr << "Could not parse batch size. Make sure you entered a valid number" << std::endl;
				std::cerr << e.what() << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		if (inputParser.cmdOptionExists("--lr")) {
			try {
				learningRate = std::stof(inputParser.getCmdOption("--lr"));
			} catch (const std::exception& e) {
				std::cerr << "Could not parse learning rate. Make sure you entered a valid number" << std::endl;
				std::cerr << e.what() << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		g_Logger = std::make_shared<Logger>();
		Trainer trainer(networkPath, batchSize, learningRate);
		trainer.train();
		return return_code;
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

		g_Logger = std::make_shared<Logger>();

		std::shared_ptr<NeuralNetwork> nn;
		if (inputParser.getCmdOption("--model") != "") {
			try {
				nn = std::make_shared<NeuralNetwork>(inputParser.getCmdOption("--model"));
			} catch (const std::exception& e) {
				std::cerr << "Could not load model. Make sure you entered a valid path" << std::endl;
				std::cerr << e.what() << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		// run in console mode
		g_IsSelfPlaying = true;
		g_Running = true;
		SelfPlay::playContinuously(nn, amount_of_sims);
		// TODO: run in threads
		
		
		/* // TODO: write code to run selfplay without GUI
		std::vector<std::thread*> threads = std::vector<std::thread*>(parallel_games, nullptr);
		for (int t = 0; t < parallel_games; t++) {
			std::thread thread_selfplay = std::thread(
				&SelfPlay::playContinuously,
				nn, // model
				amount_of_sims // amount of sims per move
			);
			threads[t] = &thread_selfplay;
		}

		LOG(DEBUG) << "Running in console mode!";
		for (int t = 0; t < parallel_games; t++) {
			if (threads[t] != nullptr) {
				threads[t]->join();
			} else {
				LOG(WARNING) << "Thread " << t << " is nullptr?";
			}
		} */
		return return_code;
	} else {
		// GUI app
		QApplication a(argc, argv);
		g_mainWindow = std::make_unique<MainWindow>();
		g_mainWindow->show();
		return_code = a.exec();
	}
	

	return return_code;
}
#include <random>

#include "game.hh"
#include "node.hh"
#include "dataset.hh"
#include "utils.hh"

Game::Game(int simulations, Environment env, Agent white, Agent black) {
	this->simulations = simulations;
	this->env = env;
	this->white = white;
	this->black = black;

	this->previous_moves = new thc::Move[2];

	// create random id
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 999999);
	
	std::string current_date = std::to_string(std::time(nullptr));
	this->game_id = "game-" + current_date + "-" + std::to_string(dis(gen));
}

void Game::reset() {
	this->env.reset();
}

int Game::playGame() {
	this->reset();

	int winner = 0;
	int counter = 0;
	while (!this->env.isGameOver()) {
		this->env.printBoard();
		
		this->play_move();
		std::cout << "Value according to white: " << this->white.getMCTS()->getRoot()->getValue() << std::endl;
		std::cout << "Value according to black: " << this->black.getMCTS()->getRoot()->getValue() << std::endl;

		counter++;
		if (counter > MAX_MOVES) {
			std::cout << "Game over by move limit" << std::endl;
			break;
		}
	}

	if (this->env.isGameOver()){
		if (this->env.terminalState == thc::TERMINAL_WCHECKMATE){
			winner = 1;
		} else if(this->env.terminalState == thc::TERMINAL_BCHECKMATE) {
			winner = -1;
		}
	}

	this->updateMemory(winner);

	this->memoryToFile();

	return winner;
}

void Game::play_move(){
	Agent* currentPlayer = this->env.getCurrentPlayer() ? &this->white : &this->black;
	std::cout << "Current player: " << currentPlayer->getName() << std::endl;

	// update mcts tree with new root
	// TODO: use child of old tree
	Node* newRoot = new Node(this->env.getFen(), nullptr, thc::Move(), 0);
	currentPlayer->updateMCTS(newRoot);

	// run the sims
	currentPlayer->getMCTS()->run_simulations(this->simulations);

	std::vector<Node*> childNodes = currentPlayer->getMCTS()->getRoot()->getChildren();

	// create memory element
	MemoryElement element;
	element.state = this->env.getFen();
	element.probs = currentPlayer->getMCTS()->getRoot()->getProbs();
	element.winner= 0;

	// save element to memory
	this->saveToMemory(element);

	// print moves
	std::cout << "Moves: " << std::endl;
	for (int i = 0; i < childNodes.size(); i++) {
		thc::Move move = childNodes[i]->getAction();
		std::cout << "Move " << i << ": " << move.NaturalOut(this->env.getRules());
		std::cout << " " << childNodes[i]->getVisitCount() << " visits" << std::endl;
	}
	
	// TODO: create distribution of moves
	/* 
	float total_probability = 0;
	for (int i = 0; i < (int)element.probs.size(); i++){
		total_probability += element.probs[i].prob;
	}
	float p = (rand() / static_cast<float>(RAND_MAX)) * total_probability;
	MoveProb* current = &element.probs[0];
	while ((p -= current->prob) > 0) {
		current++;
	} 
	*/
	
	// get move with max visit count
	if ((int)childNodes.size() == 0){
		std::cerr << "No moves available" << std::endl;
		exit(EXIT_FAILURE);
	}
	Node* current = childNodes[rand() % childNodes.size()];
	int maxVisits = 0;
	for(int i = 0; i < (int)childNodes.size(); i++){
		if (childNodes[i]->getVisitCount() >= maxVisits){
			maxVisits = childNodes[i]->getVisitCount();
			current = childNodes[i];
		}
	}

	std::cout << "Chosen move: " << current->getAction().NaturalOut(this->env.getRules()) << std::endl;
	
	// update previous moves
	this->previous_moves[0] = this->previous_moves[1];
	this->previous_moves[1] = current->getAction();

	std::cout << "Current prevmoves: " << this->previous_moves[0].src << "-" << this->previous_moves[0].dst << " and " << this->previous_moves[1].src << "-" << this->previous_moves[1].dst << std::endl;

	this->env.makeMove(current->getAction());
}

void Game::saveToMemory(MemoryElement element) {
	this->memory.push_back(element);
}

void Game::updateMemory(int winner){
	for (int i = 0; i < (int)this->memory.size(); i++){
		this->memory[i].winner = winner;
	}
}

void Game::memoryElementToTensors(MemoryElement *memory_element, torch::Tensor* input_tensor, torch::Tensor* output_tensor) {
	// convert state (string) to input (boolboards 119x8x8)
	Environment env = Environment(memory_element->state);
	torch::Tensor t = env.boardToInput();
	input_tensor = &t;
	
	// convert the probs to the policy output
	torch::Tensor policy_output = torch::zeros({73, 8, 8});
	for (MoveProb moveProb : memory_element->probs){
		thc::Move move = moveProb.move;
		std::tuple<int, int, int> plane_tuple = utils::moveToPlaneIndex(move);
		int plane = std::get<0>(plane_tuple);
		int row = std::get<1>(plane_tuple);
		int col = std::get<2>(plane_tuple);
		policy_output[plane][row][col] = moveProb.prob;
	}

	// add value to end of policy output
	torch::Tensor value_output = torch::zeros({1});
	value_output[0] = memory_element->winner;
	torch::Tensor o = torch::cat({policy_output.view({73*8*8}), value_output}, 0);
	output_tensor = &o;
}

void Game::memoryToFile(){
	std::cout << "Converting memory elements to tensors" << std::endl;
	torch::Tensor inputs = torch::zeros({(int)this->memory.size(), 119, 8, 8});
	torch::Tensor outputs = torch::zeros({(int)this->memory.size(), 73*8*8 + 1});
	for (int i = 0; i < (int)this->memory.size(); i++){
		torch::Tensor input_tensor = torch::zeros({119, 8, 8});
		torch::Tensor output_tensor = torch::zeros({73*8*8 + 1});
		memoryElementToTensors(&this->memory[i], &input_tensor, &output_tensor);
		inputs[i] = input_tensor;
		outputs[i] = output_tensor;
	}

	// create a directory for the current game
	std::cout << "Creating directory for game id = " << this->game_id << std::endl;
	std::string directory = "memory/" + this->game_id;
	if (!utils::createDirectory(directory)){
		std::cerr << "Could not create directory for current game" << std::endl;
		exit(EXIT_FAILURE);
	}

	// save tensors to file
	std::cout << "Saving tensors to file" << std::endl;
	torch::save(inputs, directory + "/input.pt");
	torch::save(outputs, directory + "/outputs.pt");

	// reset memory
	this->memory.clear();
}
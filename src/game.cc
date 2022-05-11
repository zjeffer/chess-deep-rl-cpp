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

	if(env.getCurrentPlayer()){
		this->white.updateMCTS(new Node(env.getFen(), nullptr, thc::Move(), 0.0));
	} else {
		this->black.updateMCTS(new Node(env.getFen(), nullptr, thc::Move(), 0.0));
	}

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
	this->memory.clear();
}

int Game::playGame(bool stochastic) {
	int winner = 0;
	int counter = 0;
	thc::DRAWTYPE drawType;
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
		
		if (this->env.getRules()->IsDraw(this->env.getCurrentPlayer(), drawType)){
			std::cout << "Game over by draw. Reason: " << drawType << std::endl;
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

	this->reset();

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

	if ((int)childNodes.size() == 0){
		std::cerr << "No moves available" << std::endl;
		exit(EXIT_FAILURE);
	}

	// print moves
	std::cout << "Moves: " << std::endl;
	for (int i = 0; i < childNodes.size(); i++) {
		thc::Move move = childNodes[i]->getAction();
		// std::cout << "Move " << i << ": " << move.NaturalOut(this->env.getRules());
		// std::cout << " " << childNodes[i]->getVisitCount() << " visits" << std::endl;
	}


	// TODO: if stochastic or deterministic
	thc::Move bestMove;

	if (this->stochastic){
		bestMove = getBestMoveDeterministic(element.probs);
	} else {
		bestMove = getBestMoveStochastic(element.probs);
	}	

	std::cout << "Chosen move: " << this->env.getRules()->full_move_count << ". " << bestMove.NaturalOut(this->env.getRules()) << std::endl;
	
	// update previous moves
	this->previous_moves[0] = this->previous_moves[1];
	this->previous_moves[1] = bestMove;

	// std::cout << "Current prevmoves: " << this->previous_moves[0].src << "-" << this->previous_moves[0].dst << " and " << this->previous_moves[1].src << "-" << this->previous_moves[1].dst << std::endl;

	this->env.makeMove(bestMove);
}


thc::Move Game::getBestMoveStochastic(std::vector<MoveProb> &probs){
	float total_probability = 0;
	for (int i = 0; i < (int)probs.size(); i++){
		total_probability += probs[i].prob;
	}
	float p = (rand() / static_cast<float>(RAND_MAX)) * total_probability;
	MoveProb* current = &probs[0];
	while ((p -= current->prob) > 0) {
		current++;
	}
	return current->move;
}

thc::Move Game::getBestMoveDeterministic(std::vector<MoveProb> &probs){
	// get move where probs.prob is highest
	float max_prob = 0;
	int max_index = 0;
	for (int i = 0; i < (int)probs.size(); i++){
		if (probs[i].prob > max_prob){
			max_prob = probs[i].prob;
			max_index = i;
		}
	}
	return probs[max_index].move;	
}

void Game::saveToMemory(MemoryElement element) {
	this->memory.push_back(element);
}

void Game::updateMemory(int winner){
	for (int i = 0; i < (int)this->memory.size(); i++){
		this->memory[i].winner = winner;
	}
}

void Game::memoryElementToTensors(MemoryElement *memory_element, torch::Tensor& input_tensor, torch::Tensor& output_tensor) {
	// convert state (string) to input (boolean boards 119x8x8)
	Environment env = Environment(memory_element->state);
	// flatten from [1, 119, 8, 8] to [119, 8, 8]
	input_tensor = env.boardToInput().flatten(0, 1);
	
	// convert the probs to the policy output
	torch::Tensor policy_output = torch::full({73, 8, 8}, (float)0.0);
	thc::ChessRules* rules = new thc::ChessRules();
	rules->Forsyth(memory_element->state.c_str());
	for (MoveProb moveProb : memory_element->probs){
		thc::Move move = moveProb.move;
		std::tuple<int, int, int> plane_tuple = utils::moveToPlaneIndex(move);
		policy_output[std::get<0>(plane_tuple)][std::get<1>(plane_tuple)][std::get<2>(plane_tuple)] = moveProb.prob;
	}

	// add value to end of policy output
	torch::Tensor value_output = torch::zeros({1});
	value_output[0] = memory_element->winner;
	output_tensor = torch::cat({policy_output.view({73*8*8}), value_output}, 0);
}

void Game::memoryToFile(){
	// convert MemoryElements to tensors
	std::cout << "Converting memory elements to tensors" << std::endl;
	torch::Tensor inputs = torch::zeros({(int)this->memory.size(), 119, 8, 8});
	torch::Tensor outputs = torch::zeros({(int)this->memory.size(), 73*8*8 + 1});
	for (int i = 0; i < (int)this->memory.size(); i++){
		torch::Tensor input_tensor = torch::zeros({119, 8, 8});
		torch::Tensor output_tensor = torch::zeros({73*8*8 + 1});
		memoryElementToTensors(&this->memory[i], input_tensor, output_tensor);
		inputs[i] = input_tensor.clone();
		outputs[i] = output_tensor.clone();
	}

	// create a directory for the current game
	std::cout << "Creating directory for game id = " << this->game_id << std::endl;
	std::string directory = "memory/" + this->game_id;
	if (!utils::createDirectory(directory)){
		std::cerr << "Could not create directory for current game" << std::endl;
		exit(EXIT_FAILURE);
	}

	// save tensors to file
	std::cout << "Saving tensors to files" << std::endl;
	for (int i = 0; i < (int)this->memory.size(); i++){
		// get the move number with zero padding
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << i;
		std::string move =  "/move-" + ss.str();
		// save pair to file
		torch::Tensor input = inputs[i].clone();
		torch::Tensor output = outputs[i].clone();
		if (input.numel() == 0){
			std::cerr << "Empty input tensor" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (output.numel() == 0){
			std::cerr << "Empty output tensor" << std::endl;
			exit(EXIT_FAILURE);
		}
		torch::save(inputs[i].clone(), directory + move + "-input.pt");
		torch::save(outputs[i].clone(), directory + move + "-output.pt");
	}

	// reset memory
	this->memory.clear();
}
#include "game.hh"
#include "node.hh"
#include "dataset.hh"
#include "utils.hh"
#include "common.hh"

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
	std::random_device rd = std::random_device();
	this->rng.seed(rd());
	this->dist = std::uniform_int_distribution<int>(0, RAND_MAX);

	std::uniform_int_distribution<int> game_id_dist = std::uniform_int_distribution<int>(0, 1000000);
	
	std::string current_date = std::to_string(std::time(nullptr));
	this->game_id = "game-" + current_date + "-" + std::to_string(game_id_dist(this->rng));
}

void Game::reset() {
	this->env.reset();
	this->memory.clear();
}

int Game::playGame(bool stochastic) {
	this->stochastic = stochastic;
	int winner = 0;
	int counter = 0;
	thc::DRAWTYPE drawType;
	while (!this->env.isGameOver()) {
		this->env.printBoard();
		
		this->play_move();
		G3LOG(INFO) << "Value according to white: " << this->white.getMCTS()->getRoot()->getValue();
		G3LOG(INFO) << "Value according to black: " << this->black.getMCTS()->getRoot()->getValue();

		counter++;
		if (counter > MAX_MOVES) {
			G3LOG(INFO) << "Game over by move limit";
			break;
		} 
		
		if (this->env.getRules()->IsDraw(this->env.getCurrentPlayer(), drawType)){
			G3LOG(INFO) << "Game over by draw. Reason: " << drawType;
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
	G3LOG(INFO) << "Current player: " << currentPlayer->getName();

	// update mcts tree with new root
	// TODO: use child of old tree
	Node* newRoot = new Node(this->env.getFen(), nullptr, thc::Move(), 0);
	currentPlayer->updateMCTS(newRoot);

	// run the sims
	currentPlayer->getMCTS()->run_simulations(this->simulations);

	std::vector<Node*> childNodes = currentPlayer->getMCTS()->getRoot()->getChildren();

	// TODO: add dirichlet noise to the priors of the root node's edges

	// create memory element
	MemoryElement element;
	element.state = this->env.getFen();
	element.probs = currentPlayer->getMCTS()->getRoot()->getProbs();
	element.winner= 0;

	// save element to memory
	this->saveToMemory(element);

	if ((int)childNodes.size() == 0){
		G3LOG(WARNING) << "No moves available";
		exit(EXIT_FAILURE);
	}

	// print moves
	// G3LOG(DEBUG) << "Moves: "";
	// for (int i = 0; i < childNodes.size(); i++) {
	// 	thc::Move move = childNodes[i]->getAction();
	// 	std::cout << "Move " << i << ": " << move.NaturalOut(this->env.getRules());
	// 	std::cout << " " << childNodes[i]->getVisitCount() << " visits" << std::endl;
	// }


	thc::Move bestMove;
	if (this->stochastic){
		bestMove = getBestMoveStochastic(element.probs);
	} else {
		bestMove = getBestMoveDeterministic(element.probs);
	}

	// see if move is valid
	if (!bestMove.Valid()){
		G3LOG(WARNING) << "Invalid move";
		exit(EXIT_FAILURE);
	}	

	G3LOG(INFO) << this->env.getFen();
	G3LOG(INFO) << "Chosen move: " << this->env.getRules()->full_move_count << ". " << bestMove.NaturalOut(this->env.getRules());
	
	// update previous moves
	this->previous_moves[0] = this->previous_moves[1];
	this->previous_moves[1] = bestMove;

	// G3LOG(DEBUG) << "Current prevmoves: " << this->previous_moves[0].src << "-" << this->previous_moves[0].dst << " and " << this->previous_moves[1].src << "-" << this->previous_moves[1].dst;

	this->env.makeMove(bestMove);
	thc::ILLEGAL_REASON reason;
	if (!this->env.getRules()->IsLegal(reason)){
		G3LOG(WARNING) << "Reached an illegal position after the last move. Reason: " << reason;
		this->env.printBoard();
		G3LOG(DEBUG) << this->env.getFen();
		G3LOG(DEBUG) << bestMove.src << "-" << bestMove.dst;
		G3LOG(DEBUG) << bestMove.special;
		exit(EXIT_FAILURE);
	}
}


thc::Move Game::getBestMoveStochastic(std::vector<MoveProb> &probs){
	float total_probability = 0;
	for (int i = 0; i < (int)probs.size(); i++){
		total_probability += probs[i].prob;
	}
	float p = (this->dist(this->rng) / static_cast<float>(RAND_MAX)) * total_probability;
	int index = 0;
	while ((p -= probs[index].prob) > 0) {
		index++;
	}
	return probs[index].move;
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

Environment* Game::getEnvironment(){
	return &this->env;
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
	G3LOG(DEBUG) << "Converting memory elements to tensors";
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
	G3LOG(DEBUG) << "Creating directory for game id = " << this->game_id;
	std::string directory = "memory/" + this->game_id;
	if (!utils::createDirectory(directory)){
		G3LOG(WARNING) << "Could not create directory for current game";
		exit(EXIT_FAILURE);
	}

	// save tensors to file
	G3LOG(DEBUG) << "Saving tensors to files";
	for (int i = 0; i < (int)this->memory.size(); i++){
		// get the move number with zero padding
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << i;
		std::string move =  "/move-" + ss.str();
		// save pair to file
		torch::Tensor input = inputs[i].clone();
		torch::Tensor output = outputs[i].clone();
		if (input.numel() == 0){
			G3LOG(WARNING) << "Empty input tensor";
			exit(EXIT_FAILURE);
		}
		if (output.numel() == 0){
			G3LOG(WARNING) << "Empty output tensor";
			exit(EXIT_FAILURE);
		}
		torch::save(inputs[i].clone(), directory + move + "-input.pt");
		torch::save(outputs[i].clone(), directory + move + "-output.pt");
	}

	// reset memory
	this->memory.clear();
}
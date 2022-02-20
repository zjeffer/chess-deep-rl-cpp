#include "game.hh"
#include "node.hh"
#include <algorithm>

Game::Game(Environment env, Agent white, Agent black) {
	this->env = env;
	this->white = white;
	this->black = black;

	this->previous_moves = new thc::Move[2];
}


Game::Game() : Game(Environment(), Agent("white"), Agent("black")) {
	
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
	std::cout << "Current player: " << currentPlayer->name << std::endl;

	// update mcts tree with new root
	// TODO: use child of old tree
	Node* newRoot = new Node(this->env.getFen(), nullptr, thc::Move(), 0);
	currentPlayer->updateMCTS(newRoot);

	// run the sims
	currentPlayer->getMCTS()->run_simulations(AMOUNT_OF_SIMULATIONS);

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
		std::cout << "Move " << i << ": " << move.NaturalOut(this->env.getRules()) << std::endl;
	}
	
	// create distribution of moves
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

	// for now, pick best move  (TODO)
	
	// get move with max visit count
	if ((int)childNodes.size() == 0){
		std::cerr << "No moves available" << std::endl;
		exit(1);
	}
	Node* current = childNodes[rand() % childNodes.size()];
	int maxVisits = 0;
	for(int i = 0; i < (int)childNodes.size(); i++){
		if (childNodes[i]->getVisitCount() > maxVisits){
			maxVisits = childNodes[i]->getVisitCount();
			current = childNodes[i];
		}
	}

	std::cout << "Chosen move: " << current->getAction().NaturalOut(this->env.getRules()) << std::endl;
	
	// update previous moves
	this->previous_moves[0] = this->previous_moves[1];
	this->previous_moves[1] = current->getAction();

	this->env.makeMove(current->getAction());

	std::cout << "Current prevmoves: " << this->previous_moves[0].NaturalOut(this->env.getRules()) << " and " << this->previous_moves[1].NaturalOut(this->env.getRules()) << std::endl;
}

void Game::saveToMemory(MemoryElement element) {
	this->memory.push_back(element);

}

void Game::updateMemory(int winner){
	for (int i = 0; i < (int)this->memory.size(); i++){
		this->memory[i].winner = winner;
	}
}

void Game::memoryToFile(){
	std::ofstream file;
	file.open("memory.txt");
	for (int i = 0; i < (int)this->memory.size(); i++){
		file << this->memory[i].state << " " << this->memory[i].winner << " ";
		for (int j = 0; j < (int)this->memory[i].probs.size(); j++){
			file << this->memory[i].probs[j].move.NaturalOut(this->env.getRules()) << " " << this->memory[i].probs[j].prob << " ";
		}
		file << std::endl;
	}
	file.close();
	// reset memory
	this->memory.clear();
}
#include "game.hh"
#include "common.hh"


Game::Game(int simulations, Environment* env, Agent* white, Agent* black) {
	m_Simulations = simulations;
	m_Env = env;
	m_White = white;
	m_Black = black;

	if(env->getCurrentPlayer()){
		m_White->updateMCTS(new Node(env->getFen(), nullptr, thc::Move(), 0.0));
	} else {
		m_Black->updateMCTS(new Node(env->getFen(), nullptr, thc::Move(), 0.0));
	}

	m_Previous_moves = new thc::Move[2];

	// for stochastic move selection
	m_Dist = std::uniform_int_distribution<int>(0, RAND_MAX);
	// for creating random id
	std::uniform_int_distribution<int> game_id_dist = std::uniform_int_distribution<int>(0, 1000000);
	
	// create a random id
	std::string current_date = std::to_string(std::time(nullptr));
	m_Game_id = "game-" + current_date + "-" + std::to_string(game_id_dist(g_Generator));
}

Game::~Game() {
	delete m_White;
	delete m_Black;
	delete m_Previous_moves;
}

void Game::reset() {
	m_Env->reset();
	m_Memory.clear();
}

int Game::playGame(bool stochastic) {
	m_Stochastic = stochastic;
	int winner = 0;
	int counter = 0;
	thc::DRAWTYPE drawType;
	while (!m_Env->isGameOver() && g_Running && g_IsSelfPlaying) {
		m_Env->printBoard();
		
		this->playMove();
		if (!m_Env->getCurrentPlayer()) {
			LOG(INFO) << "Value according to current player (white): " << m_White->getMCTS()->getRoot()->getQ();
		} else {
			LOG(INFO) << "Value according to current player (black): " << m_Black->getMCTS()->getRoot()->getQ();
		}

		counter++;
		if (counter > MAX_MOVES) {
			LOG(INFO) << "Game over by move limit";
			break;
		} 
		
		if (m_Env->getRules()->IsDraw(m_Env->getCurrentPlayer(), drawType)){
			m_Env->printDrawType(drawType);
			break;
		}
	}

	if (m_Env->isGameOver()){
		if (m_Env->terminalState == thc::TERMINAL_BCHECKMATE){
			winner = 1;
		} else if(m_Env->terminalState == thc::TERMINAL_WCHECKMATE) {
			winner = -1;
		}
	}

	if (!g_IsSelfPlaying || !g_Running) {
		return winner;
	}

	// only save wins
	// if (winner != 0){
	// 	this->updateMemory(winner);
	// 	this->memoryToFile();
	// }
	this->updateMemory(winner);
	this->memoryToFile();

	this->reset();

	return winner;
}

void Game::playMove(){
	Agent* currentPlayer = m_Env->getCurrentPlayer() ? m_White : m_Black;
	LOG(INFO) << "Current player: " << currentPlayer->getName();	

	// update mcts tree
	// TODO: use subtree of previously chosen move as next root
	currentPlayer->getMCTS()->setRoot(new Node(m_Env->getFen(), nullptr, thc::Move(), 0.0));

	// run the sims
	currentPlayer->getMCTS()->run_simulations(m_Simulations);

	std::vector<Node*> childNodes = currentPlayer->getMCTS()->getRoot()->getChildren();


	// create memory element
	MemoryElement element;
	element.state = m_Env->getFen();
	element.probs = currentPlayer->getMCTS()->getRoot()->getProbs();
	element.winner= 0;

	// save element to memory
	this->saveToMemory(element);

	if ((int)childNodes.size() == 0){
		LOG(WARNING) << "No moves available";
		exit(EXIT_FAILURE);
	}

	// print moves
	/* LOG(DEBUG) << "Moves: ";
	for (int i = 0; i < childNodes.size(); i++) {
		thc::Move move = childNodes[i]->getAction();
		LOG(DEBUG) << "Move " << i << ": " << move.NaturalOut(m_Env->getRules())
			<< " " << childNodes[i]->getVisitCount() << " visits. " 
			<< "PUCT score: " << childNodes[i]->getQ() << " + " << childNodes[i]->getUCB() << ". Prior: " << childNodes[i]->getPrior();
	} */
	
	thc::Move bestMove;
	if (m_Stochastic){
		bestMove = getBestMoveStochastic(element.probs);
	} else {
		bestMove = getBestMoveDeterministic(element.probs);
	}

	// see if move is valid
	if (!bestMove.Valid()){
		LOG(WARNING) << "Invalid move";
		exit(EXIT_FAILURE);
	}	

	LOG(DEBUG) << m_Env->getFen();
	LOG(INFO) << "Chosen move: " << m_Env->getRules()->full_move_count << ". " << bestMove.NaturalOut(m_Env->getRules());
	
	// update previous moves
	m_Previous_moves[0] = m_Previous_moves[1];
	m_Previous_moves[1] = bestMove;

	// LOG(DEBUG) << "Current prevmoves: " << m_Previous_moves[0].src << "-" << m_Previous_moves[0].dst << " and " << m_Previous_moves[1].src << "-" << m_Previous_moves[1].dst;

	m_Env->makeMove(bestMove);
	thc::ILLEGAL_REASON reason;
	if (!m_Env->getRules()->IsLegal(reason)){
		LOG(WARNING) << "Reached an illegal position after the last move. Reason: " << reason;
		m_Env->printBoard();
		LOG(DEBUG) << m_Env->getFen();
		LOG(DEBUG) << bestMove.src << "-" << bestMove.dst;
		LOG(DEBUG) << bestMove.special;
		exit(EXIT_FAILURE);
	}
}


thc::Move Game::getBestMoveStochastic(std::vector<MoveProb> &probs){
	// TODO: add temperature control
	float total_probability = 0;
	for (const auto& prob : probs){
		total_probability += prob.prob;
	}
	float p = (m_Dist(g_Generator) / static_cast<float>(RAND_MAX)) * total_probability;
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
	return m_Env;
}

void Game::saveToMemory(MemoryElement element) {
	m_Memory.push_back(element);
}

void Game::updateMemory(int winner){
	for (auto& element : m_Memory){
		if (winner == 0) {
			element.winner = 0;
			continue;
		}
		Environment env = Environment(element.state);
		bool winnerIsWhite = winner == 1;
		if (env.getCurrentPlayer() == winnerIsWhite) {
			// if winner is the current player, set value to 1
			element.winner = 1;
		} else {
			// if winner is not current player, set value to -1
			element.winner = -1;
		}
	}
}

void Game::memoryElementToTensors(MemoryElement *memory_element, torch::Tensor& input_tensor, torch::Tensor& output_tensor) {
	// convert state (string) to input (boolean boards 119x8x8)
	Environment env = Environment(memory_element->state);
	// flatten from [1, 119, 8, 8] to [119, 8, 8]
	input_tensor = env.boardToInput().flatten(0, 1);
	
	// convert the probs to the policy output
	torch::Tensor policy_output = torch::full({73, 8, 8}, 0.0f);
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
	torch::Tensor inputs = torch::zeros({(int)m_Memory.size(), 119, 8, 8});
	torch::Tensor outputs = torch::zeros({(int)m_Memory.size(), 73*8*8 + 1});
	for (int i = 0; i < (int)m_Memory.size(); i++){
		torch::Tensor input_tensor = torch::zeros({119, 8, 8});
		torch::Tensor output_tensor = torch::zeros({73*8*8 + 1});
		memoryElementToTensors(&m_Memory[i], input_tensor, output_tensor);
		inputs[i] = input_tensor.clone();
		outputs[i] = output_tensor.clone();
	}

	// create a directory for the current game
	LOG(DEBUG) << "Creating directory for game id = " << m_Game_id;
	std::string directory = "memory/" + m_Game_id;
	if (!utils::createDirectory(directory)){
		LOG(WARNING) << "Could not create directory for current game";
		exit(EXIT_FAILURE);
	}

	// save tensors to file
	for (int i = 0; i < (int)m_Memory.size(); i++){
		// get the move number with zero padding
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << i;
		std::string move =  "/move-" + ss.str();
		// save pair to file
		torch::Tensor input = inputs[i].clone();
		torch::Tensor output = outputs[i].clone();
		if (input.numel() == 0){
			LOG(WARNING) << "Empty input tensor";
			exit(EXIT_FAILURE);
		}
		if (output.numel() == 0){
			LOG(WARNING) << "Empty output tensor";
			exit(EXIT_FAILURE);
		}
		torch::save(inputs[i].clone(), directory + move + "-input.pt");
		torch::save(outputs[i].clone(), directory + move + "-output.pt");
	}

	// reset memory
	m_Memory.clear();
}
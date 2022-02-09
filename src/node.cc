#include "node.hh"
#include <string.h>
#include <math.h>

Node::Node(std::string fen, Node* parent, thc::Move action, float prior) {
	this->fen = fen;
	this->parent = parent;
	this->action = action;
	this->prior = prior;

	this->visit_count = 0;
	this->value = 0;
}

Node::Node() : Node("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", nullptr, thc::Move(), 1.0) {
	
}

Node::~Node() {
	// delete this->parent;
	for (int i = 0; i < (int)this->children.size(); i++) {
        delete this->children[i];
    }
}

std::string Node::getFen() {
    return this->fen;
}
void Node::setFen(std::string fen) {
    this->fen = fen;
}

Node* Node::getParent() {
    return this->parent;
}
void Node::setParent(Node* parent) {
    this->parent = parent;
}

std::vector<Node*> Node::getChildren() {
	return this->children;
}
Node* Node::getChild(std::string fen){
	for (int i = 0; i < (int)this->children.size(); i++) {
		if (strcmp(this->children[i]->getFen().c_str(), fen.c_str()) == 0) {
			return this->children[i];
		}
	}
	return nullptr;
}
void Node::add_child(Node* child) {
    this->children.push_back(child);
}


bool Node::is_leaf() {
    return this->children.size() == 0;
}

int Node::getVisitCount(){
	return this->visit_count;
}
void Node::incrementVisit() {
    this->visit_count++;
}
void Node::setVisitCount(int n){
	this->visit_count = n;
}


float Node::getValue() {
	return this->value;
}
void Node::setValue(float value) {
	this->value = value;
}

float Node::getPrior() {
	return this->prior;
}
void Node::setPrior(float prior) {
	this->prior = prior;
}

float Node::getPUCTScore(){
	if (this->getPlayer()){
		return this->getValue() + this->getUCB();
	} else {
		return (-this->getValue()) + this->getUCB();
	}
}

float Node::getQ(){
	/* TODO: add upper confidence bound */
	return this->value / (this->visit_count + 1);
}

float Node::getUCB(){
	if (this->parent == nullptr){
		return 0;
	}
	float exploration_rate = log((1 + this->parent->visit_count + 20000) / 20000) + 2;
	return exploration_rate * this->prior * (sqrt(this->parent->visit_count) / (1 + this->visit_count));
}

bool Node::getPlayer(){
	// parse the fen and return the current player
	// TODO: implement this
	return true;
}

thc::Move Node::getAction(){
	return this->action;
}
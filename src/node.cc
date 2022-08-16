#include "node.hh"
#include <cstddef>
#include <cstdint>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <iostream>
#include "common.hh"

Node::Node(std::string fen, Node* parent, thc::Move action, float prior) 
	: m_Fen(std::move(fen)), m_Parent(parent), m_Action(action), m_Prior(prior) {

}

Node::Node() {}

Node::~Node() {}


const std::string& Node::getFen() const {
    return m_Fen;
}
void Node::setFen(std::string fen) {
    m_Fen = std::move(fen);
}

Node* Node::getParent() const {
    return m_Parent;
}
void Node::setParent(Node* parent) {
    m_Parent = parent;
}

const std::vector<Node*>& Node::getChildren() const {
	return m_Children;
}

Node* Node::getChild(const std::string& fen) const {
	for (int i = 0; i < (int)m_Children.size(); i++) {
		if (strcmp(m_Children[i]->getFen().c_str(), fen.c_str()) == 0) {
			return m_Children[i];
		}
	}
	return nullptr;
}

Node* Node::getChild(const thc::Move& action) const {
	for (int i = 0; i < (int)m_Children.size(); i++) {
		if (m_Children[i]->getAction() == action) {
			return m_Children[i];
		}
	}
	return nullptr;
}

void Node::add_child(Node* child) {
    m_Children.push_back(child);
}


bool Node::isLeaf() const {
    return m_Children.size() == 0;
}

int Node::getVisitCount() const {
	return m_VisitCount;
}
void Node::incrementVisit() {
    m_VisitCount++;
}
void Node::setVisitCount(int n){
	m_VisitCount = n;
}


float Node::getValue() const {
	return m_Value;
}
void Node::setValue(float value) {
	m_Value = value;
}

float Node::getPrior() const {
	return m_Prior;
}
void Node::setPrior(float prior) {
	m_Prior = prior;
}

float Node::getPUCTScore() const{
	return getQ() + getUCB();
}

float Node::getQ() const{
	return (float)m_Value / (float)(m_VisitCount + 1);
}

float Node::getUCB() const{
	if (m_Parent == nullptr){
		LOG(WARNING) << "parent is null";
		exit(EXIT_FAILURE);
	}
	float exploration_rate = log(((float)m_Parent->getVisitCount() + 19652.0f + 1.0f) / 19652.0f) + 1.25f;
	exploration_rate *= sqrt((float)m_Parent->getVisitCount() + 1.0f) / ((float)getVisitCount() + 1.0f);
	return exploration_rate * getPrior();
}

bool Node::getPlayer() const{
	// parse the fen and return the current player
	uint8_t i = 0;
	while (m_Fen[i] != ' '){
		i++;
	}
	if(m_Fen[i+1] == 'w'){
		return true;
	} else if (m_Fen[i+1] == 'b') {
		return false;
	} else {
		LOG(FATAL) << "error: getPlayer: not white or black";
		exit(EXIT_FAILURE);
	}
}

const thc::Move& Node::getAction() const {
	return m_Action;
}

std::vector<MoveProb> Node::getProbs() const{
	std::vector<MoveProb> probs;

	for (size_t i = 0; i < m_Children.size(); i++){
		MoveProb mp;
		mp.move = m_Children[i]->getAction();
		mp.prob = (float)m_Children[i]->getVisitCount() / (float)getVisitCount();
		probs.push_back(mp);
	}

	return probs;
}
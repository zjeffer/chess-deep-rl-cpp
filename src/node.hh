#pragma once


#include "chess/thc.hh"
#include <vector>
#include <string>
#include "types.hh"


class Node {
public:
	Node(std::string fen, Node* parent, thc::Move action, float prior);
	Node();
	~Node();

	std::string getFen();
	void setFen(std::string fen);

	Node* getParent();
	void setParent(Node* parent);

	std::vector<Node*> getChildren();
	Node* getChild(std::string fen);
	Node* getChild(thc::Move action);
	
	void add_child(Node* child);

	bool isLeaf();

	int getVisitCount();
	void incrementVisit();
	void setVisitCount(int n);

	float getValue();
	void setValue(float value);

	float getPrior();
	void setPrior(float prior);

	float getPUCTScore();
	float getQ();
	float getUCB();

	bool getPlayer();

	thc::Move getAction();

	std::vector<MoveProb> getProbs();

private:
	std::string fen;
	Node* parent;
	thc::Move action;
	std::vector<Node*> children;

	int visit_count; // N
	float value; // W
	float prior; // P

};



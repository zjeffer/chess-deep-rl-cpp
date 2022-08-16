#pragma once


#include <memory>
#include <vector>
#include <string>
#include "chess/thc.hh"
#include "types.hh"


class Node {
public:
	Node(std::string fen, Node* parent, thc::Move action, float prior);
	Node();
	~Node();

	Node(const Node&);

	const std::string& getFen() const;
	void setFen(std::string fen);

	Node* getParent() const;
	void setParent(Node* parent);

	const std::vector<Node*>& getChildren() const;
	Node* getChild(const std::string& fen) const;
	Node* getChild(const thc::Move& action) const;
	
	void add_child(Node* child);

	bool isLeaf() const;

	int getVisitCount() const;
	void incrementVisit();
	void setVisitCount(int n);

	float getValue() const;
	void setValue(float value);

	float getPrior() const;
	void setPrior(float prior);

	float getPUCTScore() const;
	float getQ() const;
	float getUCB() const;

	bool getPlayer() const;

	const thc::Move& getAction() const;

	std::vector<MoveProb> getProbs() const;

private:
	std::string m_Fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
	Node* m_Parent = nullptr;
	thc::Move m_Action = thc::Move();
	std::vector<Node*> m_Children = {};

	int m_VisitCount = 0; // N
	float m_Value = 0.0f; // W
	float m_Prior = 0.0f; // P

};



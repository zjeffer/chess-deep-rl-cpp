#ifndef node_hh
#define node_hh


#include "chess/thc.hh"
#include <vector>
#include <string>

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
	void add_child(Node* child);

	bool is_leaf();

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


private:
	std::string fen;
	Node* parent;
	thc::Move action;
	std::vector<Node*> children;

	int visit_count; // N
	float value; // W
	float prior; // P

};



#endif /* node_hh */

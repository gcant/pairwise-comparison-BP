#ifndef DIGRAPH_HEADER 
#define DIGRAPH_HEADER

#include "Graph.h"

class DiGraph : public Graph {
	protected:
		std::unordered_map<int, std::unordered_set<int>> in_edges;
	public:
		DiGraph();
		int out_degree(int) const;
		int in_degree(int) const;
		std::unordered_set<int> successors(int) const;
		std::unordered_set<int> predecessors(int) const;
		void add_node(int);
		void remove_node(int);
		void add_edge(int, int);
		void remove_edge(int, int);
};

DiGraph::DiGraph(){m = 0;}

int DiGraph::out_degree(int i) const {return edge_set.at(i).size();}

int DiGraph::in_degree(int i) const {return in_edges.at(i).size();}

std::unordered_set<int> DiGraph::successors(int i) const {return edge_set.at(i);}

std::unordered_set<int> DiGraph::predecessors(int i) const {return in_edges.at(i);}

void DiGraph::add_node(int i){
	if (not(has_node(i))){
		node_set.insert(i);
		edge_set[i];
		in_edges[i];
	}
}

void DiGraph::remove_node(int i){
	m -= in_degree(i) + out_degree(i);
	node_set.erase(i);
	for (int j : edge_set[i]) in_edges[j].erase(i);
	for (int j : in_edges[i]) edge_set[j].erase(i);
	edge_set.erase(i);
	in_edges.erase(i);
}

void DiGraph::add_edge(int i, int j){
	if (not(has_edge(i,j))){
		add_node(i);
		add_node(j);
		edge_set[i].insert(j);
		in_edges[j].insert(i);
		m += 1;
	}
}

void DiGraph::remove_edge(int i, int j){
	if (has_edge(i,j)){
		edge_set[i].erase(j);
		in_edges[j].erase(i);
		m -= 1;
	}
};

#endif

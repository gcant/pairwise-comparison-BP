#ifndef GRAPH_HEADER 
#define GRAPH_HEADER

#include <unordered_set>
#include <unordered_map>

class Graph {
	protected:
		int m; //number of edges
		std::unordered_set<int> node_set;
		std::unordered_map<int, std::unordered_set<int>> edge_set;
	public:
		Graph();
		int number_of_nodes() const;
		int number_of_edges() const;
		std::unordered_set<int> nodes() const;
		std::unordered_set<int> neighbors(int) const;
		int degree(int) const;
		bool has_node(int) const;
		bool has_edge(int, int) const;
		void add_node(int);
		void remove_node(int);
		void add_edge(int, int);
		void remove_edge(int, int);
};


Graph::Graph(){m = 0;}

int Graph::number_of_nodes() const {return node_set.size();}

int Graph::number_of_edges() const {return m;}

std::unordered_set<int> Graph::nodes() const {return node_set;}

std::unordered_set<int> Graph::neighbors(int i) const {return edge_set.at(i);}

int Graph::degree(int i) const {return edge_set.at(i).size();}

bool Graph::has_node(int i) const {
	if (node_set.count(i))
		return true;
	else
		return false;
}

bool Graph::has_edge(int i, int j) const {
	if (has_node(i) && edge_set.at(i).count(j)) 
		return true;
	else
		return false;
}

void Graph::add_node(int i){
	if (not(has_node(i))){
		node_set.insert(i);
		edge_set[i];
	}
}

void Graph::remove_node(int i){
	m -= degree(i);
	node_set.erase(i);
	for (int j : edge_set[i]) edge_set[j].erase(i);
	edge_set.erase(i);
}

void Graph::add_edge(int i, int j){
	if (not(has_edge(i,j))){
		add_node(i);
		add_node(j);
		edge_set[i].insert(j);
		edge_set[j].insert(i);
		m += 1;
	}
}

void Graph::remove_edge(int i, int j){
	if (has_edge(i,j)){
		edge_set[i].erase(j);
		edge_set[j].erase(i);
		m -= 1;
	}
};

#endif

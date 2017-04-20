#ifndef net_componet_h
#define net_componet_h

#include <iostream>
 #include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <climits>
#include <cmath>
#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"

using std::vector;
using std::string;

class NeuronNetwork;
class NNLayer;
class NNWeight;
class NNNeuron;
class NNConnection;

typedef unsigned int UINT;
typedef vector<NNLayer*> VectorLayers;
typedef vector<NNWeight*>  VectorWeights;
typedef vector<NNNeuron*>  VectorNeurons;
typedef vector<NNConnection> VectorConnections;

#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))
#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )

class NeuronNetwork{
public:
	NeuronNetwork(){}
	virtual ~NeuronNetwork(){}
	void forwardPropagate(double* inputVector, UINT iCount, double* outputVector = NULL, UINT oCount = 0 );
	void backPropagate(double *actualOutput, double *desiredOutput, double LearningRate);
	VectorLayers& get(){return m_Layer;}
	void setLearningRate(double learningrate){m_LearningRate = learningrate;}
	void Serialize_Out(boost::archive::binary_oarchive& ar);
	void Serialize_In(boost::archive::binary_iarchive& ar);
private:
	VectorLayers m_Layer;
	double m_LearningRate;
};

class NNLayer{
public:
	NNLayer(const string& str,NNLayer* pPreLayer=NULL);
	virtual ~NNLayer(){}
	void forwardPropagate();
	void backPropagate(vector< double >& dErr_wrt_dXn /* in */,vector< double >& dErr_wrt_dXnm1 /* out */,double etaLearningRate);
	VectorNeurons& getNeu(){return m_Neyrons;}
	VectorWeights& getWei(){return m_Weights;}
//	template<typename Archive>
//	void Serialize_Out(Archive& ar);
//	template<typename Archive>
//	void Serialize_In(Archive& ar);
	void Serialize_Out(boost::archive::binary_oarchive& ar);
	void Serialize_In(boost::archive::binary_iarchive& ar);
private:
	string m_LayerName;
	NNLayer* pPrev;
	VectorNeurons m_Neyrons;
	VectorWeights m_Weights;
	friend class NeuronNetwork;
};

class NNWeight{
public:
	NNWeight(const string& str,double val=0.0);
	virtual ~NNWeight(){}
private:
	string m_WeiName;
	double value;
	friend class NNLayer;
};

class NNNeuron{
public:
	NNNeuron(const string& str);
	virtual ~NNNeuron(){}
	void addConnection(UINT iNeuron,UINT iWeight);
	void addConnection(const NNConnection& conn);
private:
	VectorConnections m_Connections;//本神经元输入的链接，存储对应的输入神经元与对应的权重
	string m_name;
	double output;
	friend class NeuronNetwork;
	friend class NNLayer;
};

class NNConnection{
public:
	NNConnection(UINT neuron = ULONG_MAX, UINT weight = ULONG_MAX);
	virtual ~NNConnection(){}
private:
	UINT m_Weight;
	UINT m_Neuron;
	friend class NNLayer;
};

#endif

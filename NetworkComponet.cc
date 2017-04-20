#include "NetworkComponet.h"

void NeuronNetwork::forwardPropagate(double* inputVector, UINT iCount, double* outputVector, UINT oCount){
	VectorLayers::iterator lit = m_Layer.begin();
	VectorNeurons::iterator nit;
	/*输入层直接复制*/
	if(lit < m_Layer.end()){
		nit = (*lit)->m_Neyrons.begin();
		int count = 0;
		if(iCount!=(*lit)->m_Neyrons.size())
			std::cerr<<"error in NeuronNetwork::forwardPropagate"<<std::endl;
		while((nit<(*lit)->m_Neyrons.end())&&(count < static_cast<int>(iCount))){
			(*nit)->output = inputVector[count];
			nit++;
			count++;
		}
	}
	/*对其余各层进行迭代*/
	for(lit++;lit<m_Layer.end();lit++)
		(*lit)->forwardPropagate();

	if(outputVector!=NULL){
		lit = m_Layer.end();
		lit--;
		nit = (*lit)->m_Neyrons.begin();
		for(int i=0;i<static_cast<int>(oCount);i++){
			outputVector[i] = (*nit)->output;
			nit++;
		}
	}
}

void NeuronNetwork::backPropagate(double* actualOutput, double* desiredOutput, double LearningRate){
	VectorLayers::iterator lit = m_Layer.end()-1;
	vector<double> dErr_wrt_dXlast((*lit)->m_Neyrons.size());
	vector<vector<double> > differentials;

	int iSize = m_Layer.size();
	differentials.resize(iSize);

	for(int i=0;i<static_cast<int>((*lit)->m_Neyrons.size());i++){
		dErr_wrt_dXlast[i] = actualOutput[i] - desiredOutput[i];
	}
	differentials[iSize-1] = dErr_wrt_dXlast;

	for(int i=0;i<iSize-1;i++)
		differentials[i].resize(m_Layer[i]->m_Neyrons.size(),0.0);

	for(int i=iSize-1;lit>m_Layer.begin();lit--){
		(*lit)->backPropagate(differentials[i], differentials[i-1], LearningRate);
		i--;
	}
	differentials.clear();
}


void NeuronNetwork::Serialize_In(boost::archive::binary_iarchive& ar){
	NNLayer* pLayer=NULL;
	int nLayers;

	ar >> m_LearningRate;
	ar >> nLayers;

	for(int i =0;i<nLayers;i++){
		pLayer = new NNLayer(string(""),pLayer);
		m_Layer.push_back(pLayer);
		pLayer->Serialize_In(ar);
	}
}


void NeuronNetwork::Serialize_Out(boost::archive::binary_oarchive& ar){
	VectorLayers::iterator lit;

	ar << m_LearningRate;
	ar << m_Layer.size();

	for(lit = m_Layer.begin();lit < m_Layer.end();lit++){
		(*lit)->Serialize_Out(ar);
	}
}
NNLayer::NNLayer(const string& str,NNLayer* pPreLayer):m_LayerName(str),pPrev(pPreLayer){}

void NNLayer::forwardPropagate(){
	if(!pPrev)
		std::cerr<<"error in NNLayer::forwardPropagate"<<std::endl;

	VectorNeurons::iterator nit = m_Neyrons.begin();
	VectorConnections::iterator cit;
	double sum;

	for(;nit<m_Neyrons.end();nit++){

		cit = (*nit)->m_Connections.begin();

		sum = m_Weights[(*cit).m_Weight]->value;//bias

		for(cit++;cit<(*nit)->m_Connections.end();cit++){
			sum+=m_Weights[(*cit).m_Weight]->value * pPrev->m_Neyrons[(*cit).m_Neuron]->output;
		}
		(*nit)->output = SIGMOID(sum);
	}
}

void NNLayer::backPropagate(vector< double >& dErr_wrt_dXn /* in */,vector< double >& dErr_wrt_dXnm1 /* out */,double LearningRate){
	double output;
	vector<double> dErr_wrt_dYn(m_Neyrons.size());
	vector<double> dErr_wrt_dWn;
	VectorNeurons::iterator nit = m_Neyrons.begin();
	VectorConnections::iterator cit;

	for(int i=0;i<static_cast<int>(m_Neyrons.size());i++){
		output = m_Neyrons[i]->output;
		dErr_wrt_dYn[i] = DSIGMOID(output) * dErr_wrt_dXn[i];
	}

	dErr_wrt_dWn.resize(m_Weights.size(), 0.0);
	for(int i=0;nit < m_Neyrons.end();nit++){
		cit =(*nit)->m_Connections.begin();
		for(;cit < (*nit)->m_Connections.end();cit++){
			UINT NI = (*cit).m_Neuron;
			if(ULONG_MAX==NI)
				output = 1.0;
			else
				output = pPrev->m_Neyrons[NI]->output;
			dErr_wrt_dWn[(*cit).m_Weight] += dErr_wrt_dYn[i] * output;
		}
		i++;
	}

	nit = m_Neyrons.begin();
	for(int i=0;nit < m_Neyrons.end();nit++){
		cit =(*nit)->m_Connections.begin();
		for(;cit < (*nit)->m_Connections.end();cit++){
			UINT NI = (*cit).m_Neuron;
			if(NI!=ULONG_MAX){
				dErr_wrt_dXnm1[NI] += dErr_wrt_dYn[i] * m_Weights[(*cit).m_Weight]->value;
			}
		}
		i++;
	}

	//TODO:bash gradient descent
	for (int i=0;i<static_cast<int>(m_Weights.size()); ++i ){
		double oldValue = m_Weights[i]->value;
		double newValue = oldValue - LearningRate * dErr_wrt_dWn[i];
		m_Weights[i]->value = newValue;
	}
}


void NNLayer::Serialize_In(boost::archive::binary_iarchive& ar){
	VectorNeurons::iterator nit;
	VectorWeights::iterator wit;
	VectorConnections::iterator cit;
	NNNeuron* pNeuron;
    NNWeight* pWeight;
    NNConnection conn;
	size_t iNumNeurons, iNumWeights, iNumConnections;
	string str;

	ar >> m_LayerName;
	ar >> iNumNeurons;
	ar >> iNumWeights;

	for(int i=0;i<iNumNeurons;i++){
		ar >> str;
		ar >> iNumConnections;

		pNeuron = new NNNeuron(str);
		m_Neyrons.push_back(pNeuron);

		for(int j=0;j<iNumConnections;j++){
			ar >> conn.m_Neuron;
			ar >> conn.m_Weight;
			pNeuron->addConnection(conn);
		}
	}

	for(int i=0;i<iNumWeights;i++){
		ar >> str;
		pWeight = new NNWeight(str);

		ar >> pWeight->value;

		m_Weights.push_back(pWeight);
	}
}


void NNLayer::Serialize_Out(boost::archive::binary_oarchive& ar){
	VectorNeurons::iterator nit;
	VectorWeights::iterator wit;
	VectorConnections::iterator cit;

	ar << m_LayerName;
	ar << m_Neyrons.size();
	ar << m_Weights.size();

	for(nit = m_Neyrons.begin();nit<m_Neyrons.end();nit++){
		NNNeuron& n = *(*nit);
		ar << n.m_name;
		ar <<n.m_Connections.size();
		for(cit = n.m_Connections.begin();cit<n.m_Connections.end();cit++){
			ar << (*cit).m_Neuron;
			ar << (*cit).m_Weight;
		}
	}

	for(wit = m_Weights.begin();wit<m_Weights.end();wit++){
		ar << (*wit)->m_WeiName;
		ar << (*wit)->value;
	}
}

NNNeuron::NNNeuron(const string& str):m_name(str),output(0.0){}

void NNNeuron::addConnection(UINT iNeuron, UINT iWeight){
	m_Connections.push_back(NNConnection(iNeuron,iWeight));
}

void NNNeuron::addConnection(const NNConnection& conn){
	m_Connections.push_back(conn);
}


NNWeight::NNWeight(const string& str, double val):m_WeiName(str),value(val){}

NNConnection::NNConnection(UINT neuron, UINT weight):m_Weight(weight),m_Neuron(neuron){}

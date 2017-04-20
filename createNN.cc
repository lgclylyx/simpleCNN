#include "createNN.h"
namespace{
	template<typename T>
	string to_string(T num){
		std::stringstream trans;
		string re;
		trans<<num;
		trans>>re;
		return re;
	}
	size_t swap_endian(size_t val){//分治，先将高16位和低16位分别置换，然后再将两个16位总的进行置换
		val = ((val << 8)&0xFF00FF00) | ((val >> 8)&0x00FF00FF);
		return (val << 16) | (val >> 16);
	}
};

NN::NN(){

}

NN::~NN(){}

void NN::init_train(){
	NNLayer* pLayer;

	//layer0  input
	pLayer = new NNLayer("Layer00_input");
	m_NN.get().push_back(pLayer);
	for(int i=0;i<841;i++){
		string NeurName="Layer00_Neur_Num";
		NeurName=NeurName+to_string(i);
		pLayer->getNeu().push_back(new NNNeuron(NeurName));
	}

	//layer1 Conv
	//每个链接的权重被拉成的向量，对于然后在根据网络结构设置相应的链接，将权重和神经元之间的链接匹配
	pLayer = new NNLayer("Layer01_conv1",pLayer);
	m_NN.get().push_back(pLayer);
	for (int i=0; i<1014; ++i){
		string NeurName="Layer01_Neur_Num";
		NeurName=NeurName+to_string(i);
		pLayer->getNeu().push_back(new NNNeuron(NeurName));
	}
	for(int i=0;i<156;i++){
		string NeurName="Layer01_Wei_Num";
		NeurName=NeurName+to_string(i);
		pLayer->getWei().push_back(new NNWeight(NeurName,(0.05 * UNIFORM_PLUS_MINUS_ONE)));
	}
    int kernelTemplate[25] = {//后一层的神经元在前一幅图片上的要链接的神经元的相对位置
        0,  1,  2,  3,  4,
        29, 30, 31, 32, 33,
        58, 59, 60, 61, 62,
        87, 88, 89, 90, 91,
        116,117,118,119,120 };
    for(int fm=0;fm<6;fm++){
    	for(int i=0;i<13;i++){
    		for(int j=0;j<13;j++){
    			int iNumWeight = fm * 26;
    			NNNeuron& n = *( pLayer->getNeu()[j+13*i+169*fm]);
    			n.addConnection(ULONG_MAX, iNumWeight++);//bias,对于每一个神经元来说对于前一层链接，并且有一个偏置，且偏置的输出是1，根据权重改变大小
    			for(int k=0;k<25;k++){
    				n.addConnection(2*j+i*58+kernelTemplate[k], iNumWeight++);
    			}
    		}
    	}
    }

    //layer2 Conv
    pLayer = new NNLayer("Layer02_conv2",pLayer);
    m_NN.get().push_back(pLayer);
    for (int i=0; i<1250; ++i){
        string NeurName="Layer02_Neur_Num";
        NeurName=NeurName+to_string(i);
        pLayer->getNeu().push_back(new NNNeuron(NeurName));
    }
    for(int i=0;i<7800;i++){
        string NeurName="Layer02_Wei_Num";
        NeurName=NeurName+to_string(i);
        pLayer->getWei().push_back(new NNWeight(NeurName,(0.05 * UNIFORM_PLUS_MINUS_ONE)));
    }
    int kernelTemplate2[25] = {
        0,  1,  2,  3,  4,
        13, 14, 15, 16, 17,
        26, 27, 28, 29, 30,
        39, 40, 41, 42, 43,
        52, 53, 54, 55, 56   };
    for(int fm=0;fm<50;fm++){
        for(int i=0;i<5;i++){
        	for(int j=0;j<5;j++){
        	    int iNumWeight = fm * 26;//每一个feature上的每一个神经元得权值相等
        	    NNNeuron& n = *( pLayer->getNeu()[j+5*i+25*fm]);//在每个feature上从左自右，从上自下，遍历每一个神经元设置链接和共享的权重
        	    n.addConnection(ULONG_MAX, iNumWeight++);//bias
        	    for(int k=0;k<25;k++){
        	    	n.addConnection(2*j+i*26+kernelTemplate2[k], iNumWeight++);
        	    	n.addConnection(169+2*j+i*26+kernelTemplate2[k], iNumWeight++);
        	    	n.addConnection(338+2*j+i*26+kernelTemplate2[k], iNumWeight++);
        	    	n.addConnection(507+2*j+i*26+kernelTemplate2[k], iNumWeight++);
        	    	n.addConnection(676+2*j+i*26+kernelTemplate2[k], iNumWeight++);
        	    	n.addConnection(845+2*j+i*26+kernelTemplate2[k], iNumWeight++);
        	    }
        	 }
        }
    }

    //layer3 full-connect
    pLayer = new NNLayer("Layer03_full_connect",pLayer);
    m_NN.get().push_back(pLayer);
    for(int i=0; i<100; ++i){
       string NeurName="Layer03_Neur_Num";
       NeurName=NeurName+to_string(i);
       pLayer->getNeu().push_back(new NNNeuron(NeurName));
    }
    for(int i=0;i<125100;i++){
       string NeurName="Layer03_Wei_Num";
       NeurName=NeurName+to_string(i);
       pLayer->getWei().push_back(new NNWeight(NeurName,(0.05 * UNIFORM_PLUS_MINUS_ONE)));
    }
    for(int fm=0;fm<100;fm++){
    	int iNumWeight = fm * 1251;
    	NNNeuron& n = *(pLayer->getNeu()[fm]);
    	n.addConnection(ULONG_MAX,iNumWeight++);
    	for(int i=0;i<1250;i++)
    		n.addConnection(i, iNumWeight++);
    }

    //layer4 output
    pLayer = new NNLayer("Layer04_output",pLayer);
    m_NN.get().push_back(pLayer);
    for(int i=0; i<10; ++i){
       string NeurName="Layer04_Neur_Num";
       NeurName=NeurName+to_string(i);
       pLayer->getNeu().push_back(new NNNeuron(NeurName));
    }
    for(int i=0;i<1010;i++){
       string NeurName="Layer04_Wei_Num";
       NeurName=NeurName+to_string(i);
       pLayer->getWei().push_back(new NNWeight(NeurName,(0.05 * UNIFORM_PLUS_MINUS_ONE)));
    }
    for(int fm=0;fm<10;fm++){
       int iNumWeight = fm * 101;
       NNNeuron& n = *(pLayer->getNeu()[fm]);
       n.addConnection(ULONG_MAX,iNumWeight++);
       for(int i=0;i<100;i++)
          n.addConnection(i, iNumWeight++);
    }
}

void NN::init_test(const string& filename){
	std::ifstream file(filename.c_str(),std::ios::binary);
	boost::archive::binary_iarchive ar(file);
	m_NN.Serialize_In(ar);
}

//TODO:1.多线程改造，用多线程将前向后向传播同时进行；（保存每个神经元的前向输出，以供后向使用）2。对图片进行变换；3.对误差较小的输出不反向传播
void NN::train_proc(const string& str,const string& filename){
	if(str.compare("start from mid")==0){
		init_test(filename);
	}else{
		init_train();
	}

	m_NN.setLearningRate(0.001);

	int batch_Num = 60000;
	size_t categ = 10;
	size_t epoch = 9;

	std::ifstream trainImageFile("./data/train-images.idx3-ubyte",std::ios::in|std::ios::binary);
	std::ifstream trainLabelFile("./data/train-labels.idx1-ubyte",std::ios::in|std::ios::binary);

	dataset data_param;

	trainImageFile.read(reinterpret_cast<char*>(&data_param.magic),4);
	data_param.magic = swap_endian(data_param.magic);
	trainLabelFile.read(reinterpret_cast<char*>(&data_param.magic),4);
	data_param.magic = swap_endian(data_param.magic);

	trainImageFile.read(reinterpret_cast<char*>(&data_param.num_images),4);
	data_param.num_images = swap_endian(data_param.num_images);

	trainLabelFile.read(reinterpret_cast<char*>(&data_param.num_labels),4);
	data_param.num_labels = swap_endian(data_param.num_labels);

	trainImageFile.read(reinterpret_cast<char*>(&data_param.num_rows),4);
	data_param.num_rows = swap_endian(data_param.num_rows);
	trainImageFile.read(reinterpret_cast<char*>(&data_param.num_cols),4);
	data_param.num_cols = swap_endian(data_param.num_cols);

	if(batch_Num > static_cast<int>(data_param.num_images))
		batch_Num = data_param.num_images;

	unsigned char* inputdata = new unsigned char[batch_Num*(data_param.num_rows+1)*(data_param.num_cols+1)]();
	unsigned char* inputlabel = new unsigned char[batch_Num]();

	double* inputVector = new double[batch_Num*(data_param.num_rows+1)*(data_param.num_cols+1)]();
	int* label = new int[batch_Num];
	double* targetOutputVector = new double[categ*batch_Num]();
	double* actualOutputVector = new double[categ*batch_Num];

	for(int i=0;i<static_cast<int>(batch_Num*(data_param.num_rows+1)*(data_param.num_cols+1));i++)
		inputVector[i] = 1.0;
	for(int i=0;i<static_cast<int>(categ*batch_Num);i++)
		targetOutputVector[i] = -1.0;
	while((--epoch)>=0){
		string NnName="NN_";
		NnName=NnName+to_string(epoch)+string(".nn");
		std::ofstream file(NnName.c_str(),std::ios::binary|std::ios::trunc);
		boost::archive::binary_oarchive oa(file);

		std::cout<<"the "<<epoch<<" epochs remaining....."<<std::endl;
		clock_t start,end;
		start = clock();

		trainImageFile.read(reinterpret_cast<char*>(inputdata), batch_Num*data_param.num_rows*data_param.num_cols);
		trainLabelFile.read(reinterpret_cast<char*>(inputlabel), batch_Num);

		for(int k=0;k<batch_Num;k++)
			for(int i=1;i<static_cast<int>(data_param.num_rows);i++){
				for(int j=0;j<static_cast<int>(data_param.num_cols);j++){
					inputVector[1 + j + 29*i+k*(data_param.num_rows+1)*(data_param.num_cols+1)] = (double)((int)(unsigned char)inputdata[ j + data_param.num_rows*i+k*data_param.num_rows*data_param.num_cols])/128.0 - 1.0;
				}
			}
		for(int i = 0;i<batch_Num;i++){
			label[i] = inputlabel[i];
			if(label[i] < 0)
				label[i] = 0;
			else if(label[i] > 9)
				label[i] = 9;
		}
		for(int i = 0;i<batch_Num;i++)
			targetOutputVector[i*categ+label[i]]=1.0;

		for(int i=0;i<batch_Num;i++){
			m_NN.forwardPropagate(inputVector+i*(data_param.num_rows+1)*(data_param.num_cols+1), (data_param.num_rows+1)*(data_param.num_cols+1), actualOutputVector+i*categ,categ);
			m_NN.backPropagate(actualOutputVector+i*categ, targetOutputVector+i*categ, 0.001);
		}
		end = clock();
		std::cout<<"times: "<<(double)(end - start)/CLOCKS_PER_SEC<<std::endl;
		m_NN.Serialize_Out(oa);
		file.close();
	}
	delete [] inputdata;
	delete [] inputlabel;
	delete [] inputVector;
	delete [] label;
	delete [] targetOutputVector;
	delete [] actualOutputVector;
}

void NN::test_proc(const string& filename){
	init_test(filename);

	size_t categ = 10;
	int batch_Num = 10000;
	int* iBestIndex = new int [batch_Num];
	size_t correc_Num = 0;

	std::ifstream testImageFile("./data/t10k-images.idx3-ubyte",std::ios::in|std::ios::binary);
	std::ifstream testLabelFile("./data/t10k-labels.idx1-ubyte",std::ios::in|std::ios::binary);

	dataset data_param;
	testImageFile.read(reinterpret_cast<char*>(&data_param.magic),4);
	data_param.magic = swap_endian(data_param.magic);
	testLabelFile.read(reinterpret_cast<char*>(&data_param.magic),4);
	data_param.magic = swap_endian(data_param.magic);

	testImageFile.read(reinterpret_cast<char*>(&data_param.num_images),4);
	data_param.num_images = swap_endian(data_param.num_images);

	testLabelFile.read(reinterpret_cast<char*>(&data_param.num_labels),4);
	data_param.num_labels = swap_endian(data_param.num_labels);

	testImageFile.read(reinterpret_cast<char*>(&data_param.num_rows),4);
	data_param.num_rows = swap_endian(data_param.num_rows);
	testImageFile.read(reinterpret_cast<char*>(&data_param.num_cols),4);
	data_param.num_cols = swap_endian(data_param.num_cols);

	if(batch_Num > static_cast<int>(data_param.num_images))
		batch_Num = data_param.num_images;

	unsigned char* inputdata = new unsigned char[batch_Num*(data_param.num_rows+1)*(data_param.num_cols+1)]();
	unsigned char* inputlabel = new unsigned char[batch_Num]();

	double* inputVector = new double[batch_Num*(data_param.num_rows+1)*(data_param.num_cols+1)]();
	int* label = new int[batch_Num];
	double* actualOutputVector = new double[categ*batch_Num];

	for(int i=0;i<static_cast<int>(batch_Num*(data_param.num_rows+1)*(data_param.num_cols+1));i++)
		inputVector[i] = 1.0;

	testImageFile.read(reinterpret_cast<char*>(inputdata), batch_Num*data_param.num_rows*data_param.num_cols);
    testLabelFile.read(reinterpret_cast<char*>(inputlabel), batch_Num);

	for(int k=0;k<batch_Num;k++)
		for(int i=1;i<static_cast<int>(data_param.num_rows);i++){
			for(int j=0;j<static_cast<int>(data_param.num_cols);j++){
				inputVector[1 + j + 29*i+k*(data_param.num_rows+1)*(data_param.num_cols+1)] = (double)((int)(unsigned char)inputdata[ j + data_param.num_rows*i+k*data_param.num_rows*data_param.num_cols])/128.0 - 1.0;
			}
		}
	for(int i = 0;i<batch_Num;i++){
		label[i] = inputlabel[i];
		if(label[i] < 0)
			label[i] = 0;
		else if(label[i] > 9)
			label[i] = 9;
	}

	for(int i=0;i<batch_Num;i++){
		std::cout <<"the "<<i<<" turn....."<<std::endl;
		m_NN.forwardPropagate(inputVector+i*(data_param.num_rows+1)*(data_param.num_cols+1), (data_param.num_rows+1)*(data_param.num_cols+1), actualOutputVector+i*categ,categ);
	}

	for(int i=0;i<batch_Num;i++){
		double maxValue = -99.0;
		for(int j=0;j<static_cast<int>(categ);j++){
			if(actualOutputVector[j+i*categ]>maxValue){
				iBestIndex[i] = j;
				maxValue = actualOutputVector[j+i*categ];
			}
		}
	}

	for(int i = 0;i < batch_Num;i++){
		std::cout<<"the "<<i<<" turn test-> target: "<<label[i]<<" actual: "<<iBestIndex[i];
		if(label[i] == iBestIndex[i]){
			correc_Num++;
			std::cout<<"	right!"<<std::endl;
		}else
			std::cout<<"	wrong!"<<std::endl;
	}
	std::cout<<"error rate: "<<std::fixed<<std::setprecision(2)<<(100*(data_param.num_images-correc_Num)/data_param.num_images)<<"%"<<std::endl;
	delete [] inputdata;
	delete [] inputlabel;
	delete [] inputVector;
	delete [] label;
	delete [] actualOutputVector;
	delete [] iBestIndex;
}

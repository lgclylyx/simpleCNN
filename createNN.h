#ifndef create_NN_h
#define create_NN_h

#include "NetworkComponet.h"
#include <sstream>
#include <ctime>

class NN{
private:
	struct dataset{
		size_t magic;
		size_t num_images;
		size_t num_labels;
		size_t num_rows;
		size_t num_cols;
	};
public:
	NN();
	~NN();
	void train_proc(const string& str="",const string& filename = "");
	void test_proc(const string& filename);
private:
	void init_train();
	void init_test(const string& filename);
	NeuronNetwork m_NN;

};

#endif

#include "framework.h"

void framework::sequential::add(layer* l)
{
	totalMemorySize += (l->getMemSize());
	seq.push_back(l);
}

framework::layer* framework::sequential::operator[](int i)
{
	return seq[i];
}

void framework::sequential::inference(float* x, float* y)
{
	seq[0]->setMem(x);
	seq.back()->setMem(y);

	for (int i = 0; i < seq.size(); ++i) {
		seq[i]->run();
	}
}

bool framework::sequential::loadWeight(std::vector<float*> weights)
{
	if (weights.size() > seq.size())
		return false;

	for (int i = 1; i < seq.size(); ++i)
	{
		Serial.printf("Loading weight : %s\n", seq[i]->getName().c_str());

		int layerParamSize = seq[i]->getNParam();
		if (layerParamSize != 0) // Some layers don't need weight
			seq[i]->setParam(weights[(i - 1)]); // TODO : Check if weight with bias work
	}
	return true;
}

#ifdef FOR_ARDUINO
void framework::sequential::summary()
{
	size_t nParamAll = 0;
	Serial.printf("\n\nModel Summary\n\n");
	Serial.printf("#\tLayer\t\tOutput Shape[N,C,H,W]\t\tParameters\n");
	for (int i = 0; i < seq.size(); ++i)
	{
		Serial.printf("--------------------------------------------------------------------\n");
		shape_t s = seq[i]->getShape();
		std::string n = seq[i]->getName();
		size_t nParam = seq[i]->getNParam();
		nParamAll += nParam;
		Serial.printf("[%d]\t%-10s\t[%3d, %3d, %3d, %3d]\t\t%zu\n", i, n.c_str(), s.N, s.C, s.H, s.W, nParam);
	}
	Serial.printf("--------------------------------------------------------------------\n\n");
	Serial.printf("                                                        %zu\n\n", nParamAll);
}

#else
void framework::sequential::summary()
{
	size_t nParamAll = 0;
	printf("\n\nModel Summary\n\n");
	printf("#\tLayer\t\tOutput Shape[N,C,H,W]\t\tParameters\n");
	for (int i = 0; i < seq.size(); ++i)
	{
		printf("--------------------------------------------------------------------\n");
		shape_t s = seq[i]->getShape();
		std::string n = seq[i]->getName();
		size_t nParam = seq[i]->getNParam();
		nParamAll += nParam;
		printf("[%d]\t%-10s\t[%3d, %3d, %3d, %3d]\t\t%zu\n", i, n.c_str(), s.N, s.C, s.H, s.W, nParam);
	}
	printf("--------------------------------------------------------------------\n\n");
	printf("                                                        %zu\n\n", nParamAll);
}
#endif

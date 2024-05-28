#include "framework.h"

void ReLU(size_t memSize, float* prvMEM, float* MEM) 
{
    for (int i = 0; i < memSize; ++i)
        MEM[i] = std::max(0.0f, prvMEM[i]);
}

framework::relu::relu(std::string name, layer* prvLayer)
{
    this->name = name;
    this->prvLayer = prvLayer;
    shape = prvLayer->getShape();
    memSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);

    setNParam(0);
}

void framework::relu::run() 
{
    if (MEM == nullptr)
        MEM = new float[memSize];
    float* prvMEM = prvLayer->getMem();
    ReLU(memSize, prvMEM, MEM);
}

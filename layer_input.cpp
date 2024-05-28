#include "framework.h"

framework::input::input(shape_t s)
{
	name = "Input";
	prvLayer = nullptr;
	shape = s;
	memSize = s.N * s.C * s.H * s.W * sizeof(float);
	MEM = new float[memSize];
  setNParam(0);
}

void framework::input::run() { }

size_t framework::input::setParam(float* buffer) { return 0; }
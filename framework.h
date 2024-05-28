/**
* ESP32 Ready for porting version
* 04/24/2023
*
* OpenMP & SIMD version
* Pending
*/

#pragma once

#define FOR_ARDUINO

#ifdef FOR_ARDUINO
#include <Arduino.h>
#include <WiFi.h>
extern bool frameworkTaskFlag;

#else
#include <omp.h>

#endif

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <limits>
#include <string>
#include <cstring>

#include <inttypes.h>

using shape_t = struct shape
{
	int N;
	int C;
	int H;
	int W;
};

namespace framework
{
	class layer
	{
	protected:
		std::string name;
		float* MEM = nullptr;
		layer* prvLayer = nullptr;
		shape_t shape;
		shape_t prvShape;
		size_t nParam = 0;
		size_t memSize = 0; // output memory size to a layer

	public:
		layer() {};
		~layer() {};
		virtual void run() = 0;
		virtual size_t setParam(float* buffer) = 0; // todo : test if can modification to shared pointer

		void setMem(float* m) { MEM = m; };
		float* getMem() { return MEM; };

		size_t getMemSize() { return memSize; };

		shape_t setName(const std::string& n) { name = n; };
		std::string getName() { return name; };

		void setPrvLayer(layer* p) { prvLayer = p; };
		layer* getPrvLayer() { return prvLayer; };

		void setShape(const shape_t& s) { shape = s; };
		shape_t getShape() { return shape; };

		void setNParam(const size_t& n) { nParam = n; };
		size_t getNParam() { return nParam; };
	};


	class input : public layer
	{
	public:
		input() = delete;
		input(const input&) = delete;
		input(shape_t s);
		~input() {};
		void run();
		size_t setParam(float* buffer);
	};


	class conv2d : public layer
	{
	private:
		float* wMEM = nullptr;
		float* bMEM = nullptr;
		bool useBias;
		int padSize;
		int stride;
		int kSize;
		size_t weightSize;
		size_t biasSize;

	public:
		conv2d() = delete;
		conv2d(const conv2d&) = delete;
		conv2d(std::string name, layer* p, int cout, int ks, int stride = 1, int pad = 0, bool use_bias = false);
		~conv2d() {};
		void run();
		size_t setParam(float* buffer);
	};


	class relu : public layer
	{
	public:
		relu() = delete;
		relu(const relu&) = delete;
		relu(std::string name, layer* p);
		void run();
		size_t setParam(float* buffer) { return 0; };
	};


	class sequential
	{
	private:
		std::vector<layer*> seq;
		size_t totalMemorySize = 0;
	public:
		// Expose to user
		// vector should be loaded with layer order
		std::vector<float*> modelWeights;

		sequential() {}
		~sequential() {}

		void add(layer* l);
		size_t getMemSize() { return totalMemorySize; }
		layer* operator[](int i);
		layer* back() { return seq.back(); }
		void inference(float* x, float* y);
		bool loadWeight(std::vector<float*> weights);
		void summary();
	};

}
#include "framework.h"

void Conv2D(int inputH, int inputW, int inChannel, int outputH, int outputW, int outChannel, int batchSize, int stride, int kSize, int padSize, float* X, float* W, float* B, float* Y)
{
	int wSubSize = kSize * kSize * inChannel;
	int xOneBatchSize = inputH * inputW * inChannel;
	int xMapSize = inputH * inputW;
	int yOneBatchSize = outputH * outputW * outChannel;
	int yMapSize = outputH * outputW;

	float sum;
	int indexW;

	for (int N = 0; N < batchSize; ++N)
	{
		for (int h = 0; h < outputH; ++h)
		{
			for (int w = 0; w < outputW; ++w)
			{
				for (int cOut = 0; cOut < outChannel; ++cOut)
				{
					sum = 0.0f;
					indexW = 0;
					for (int cIn = 0; cIn < inChannel; ++cIn)
					{
						for (int kh = 0; kh < kSize; ++kh)
						{
							for (int kw = 0; kw < kSize; ++kw, ++indexW)
							{
								int hp = h * stride + kh - padSize;
								int wp = w * stride + kw - padSize;
								if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW)
									sum += W[cOut * wSubSize + indexW] * X[N * xOneBatchSize + cIn * xMapSize + hp * inputW + wp];
							}
						}
					}
					if (B != nullptr) {
						Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w] = sum + B[cOut];
					}
					else {
						Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w] = sum;
					}
				}
			}
		}
	}
}

framework::conv2d::conv2d(std::string name, layer* prvLayer, int cout, int kSize, int stride, int padSize, bool useBias)
{
	this->name = name;
	this->useBias = useBias;
	this->padSize = padSize;
	this->stride = stride > 2 ? 2 : stride;
	this->kSize = kSize;
	this->prvLayer = prvLayer;
	prvShape = prvLayer->getShape();
	shape.N = prvShape.N;
	shape.C = cout;
	shape.H = (prvShape.H - this->kSize + 2 * this->padSize) / this->stride + 1;
	shape.W = (prvShape.W - this->kSize + 2 * this->padSize) / this->stride + 1;

	weightSize = shape.C * prvShape.C * this->kSize * this->kSize;
	biasSize = shape.C;
	setNParam(weightSize + static_cast<int>(this->useBias) * this->biasSize);
	memSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);
}


void framework::conv2d::run()
{
	if (MEM == nullptr)
		MEM = new float[memSize];
	float* prvMEM = prvLayer->getMem();
	Conv2D(prvShape.H, prvShape.W, prvShape.C, shape.H, shape.W, shape.C, shape.N, stride, kSize, padSize, prvMEM, wMEM, bMEM, MEM);
}

/* Scheme can be improved */
size_t framework::conv2d::setParam(float* buffer)
{
	wMEM = new float[weightSize];
	memcpy(wMEM, buffer, weightSize * sizeof(float));

	if (useBias) {
		bMEM = new float[biasSize];
		memcpy(bMEM, buffer + weightSize, biasSize * sizeof(float));
	}

	return weightSize + static_cast<int>(useBias) * biasSize;
}
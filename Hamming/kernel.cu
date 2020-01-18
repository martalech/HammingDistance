#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>
#include <cmath>
#include <thrust/device_vector.h>

constexpr unsigned long long kNumberOfBits = 16;
constexpr unsigned long long kNumberOfSequences = 1024;
constexpr unsigned long long kNumberOfPairs = (kNumberOfSequences * (kNumberOfSequences - 1)) / 2;

class Sequence
{
public:
	char& operator[](unsigned long long int i) {
		return bytes[i];
	}
	char bytes[(kNumberOfBits / 64 + (!!(kNumberOfBits % 64))) * 8];
};

void generateInput(Sequence* bits);
__host__ __device__ unsigned long long* getWord(char* bits, unsigned long long j);
__host__ __device__ char checkDistance(Sequence* sequence1, Sequence* sequence2, unsigned long long nrOfBits);
__host__ __device__ inline void k2ij(unsigned long long  k, unsigned long long* i, unsigned long long* j);
__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j);
__global__ void hammingGPU(Sequence* d_sequences, int* results, unsigned long long nrOfSeq, unsigned long long nrOfBits);

__global__ void hammingGPU(Sequence* d_sequences, int* results, unsigned long long nrOfSeq, unsigned long long nrOfBits) {
	unsigned long long threadId = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned long long i, j;
	//k2ij(threadId, &j, &i);
	for (int i = threadId + 1; i < nrOfSeq; i++) {
		int ret = checkDistance(d_sequences + threadId, d_sequences + i, nrOfBits);
		results[ij2k(i, threadId)] = ret;
	}
}

int pairsGPU(Sequence* h_sequence) {
	Sequence* d_sequence;
	int *h_results, *d_results;
	h_results = new int[kNumberOfPairs];
	//for (int i = 0; i < kNumberOfSequences; i++) {
	//	for (int j = i + 1; j < kNumberOfSequences; j++) {
	//		std::cout << "ij2k: " << ij2k(j, i) << std::endl;
	//	}
	//}
	cudaMalloc(&d_sequence, sizeof(Sequence) * kNumberOfSequences);
	cudaMemcpy(d_sequence, h_sequence, sizeof(Sequence) * kNumberOfSequences, cudaMemcpyHostToDevice);
	cudaMalloc(&d_results, sizeof(int) * kNumberOfPairs);
	cudaMemcpy(d_results, h_results, sizeof(int) * kNumberOfPairs, cudaMemcpyHostToDevice);
	hammingGPU << < kNumberOfSequences / 1024, kNumberOfSequences >> > (d_sequence, d_results, kNumberOfSequences, kNumberOfBits);
	if (kNumberOfSequences % 1024)
	{
		hammingGPU << < 1, kNumberOfSequences % 1024 >> > (d_sequence, d_results, kNumberOfSequences, kNumberOfBits);
	}
	cudaMemcpy(h_results, d_results, sizeof(int) * kNumberOfPairs, cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < kNumberOfPairs; i++) {
		sum += h_results[i];
	}
	std::cout << "sum gpu: " << sum;
	return 0;
}

int pairsCPU(Sequence* sequences) {
	Sequence* d_sequence;
	int *results;
	results = new int[kNumberOfPairs];
	//for (int i = 0; i < kNumberOfSequences; i++) {
	//	for (int j = i + 1; j < kNumberOfSequences; j++) {
	//		std::cout << "ij2k: " << ij2k(j, i) << std::endl;
	//	}
	//}
	for (int i = 0; i < kNumberOfSequences; i++) {
		for (int j = i + 1; j < kNumberOfSequences; j++) {
			int ret = checkDistance(sequences + i, sequences + j, kNumberOfBits);
			results[ij2k(j, i)] = ret;
		}
	}
	int sum = 0;
	for (int i = 0; i < kNumberOfPairs; i++) {
		sum += results[i];
	}
	std::cout << "sum cpu: " << sum;
	return 0;
}

int main() {
	Sequence* sequences = new Sequence[kNumberOfSequences];
	generateInput(sequences);
	pairsCPU(sequences);
	pairsGPU(sequences);
    return 0;
}

void generateInput(Sequence* bits) {
	std::mt19937_64 random;
	int seed = std::random_device()();
	random.seed(seed);
	for (int i = 0; i < kNumberOfSequences; i++) {
		for (int j = 0; j < kNumberOfBits / 64; j++) {
			*getWord(bits[i].bytes, j) = random();
		}
		*getWord(bits[i].bytes, kNumberOfBits / 64) = random() >> (64 - (kNumberOfBits % 64));
	}
}

__host__ __device__ unsigned long long* getWord(char* bits, unsigned long long j) {
	return (unsigned long long*)(bits + j * 64 / 8);
}

__host__ __device__ char checkDistance(Sequence* sequence1, Sequence* sequence2, unsigned long long nrOfBits)
{
	int diff = 0;
	for (int j = 0; j < (nrOfBits + 63) / 64; ++j)
	{
		unsigned long long int a, b, xor;
		a = *(getWord(sequence1->bytes, j));
		b = *(getWord(sequence2->bytes, j));
		xor = a ^ b;
		diff += xor == 0 ? 0 : (xor & (xor -1) ? 2 : 1);
		if (diff > 1)
		{
			return 0;
		}
	}
	return !!diff;
	//return 0;
}

__host__ __device__ inline void k2ij(unsigned long long  k, unsigned long long* i, unsigned long long* j)
{
	//adding 1 to k to skip first result
	*i = (unsigned int)(0.5 * (-1 + sqrtl(1 + 8 * (k + 1))));
	//decreasing 1 from j , as we start from 0 not 1
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j)
{
	return ((unsigned long long)i) * (i - 1) / 2 + j;
}

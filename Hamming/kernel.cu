#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>

constexpr unsigned long long kNumberOfBits = 10;
constexpr unsigned long long kNumberOfSequences = 10000;
constexpr unsigned long long kNumberOfPairs = (kNumberOfSequences * (kNumberOfSequences - 1)) / 2;

class Sequence {
	char bytes[(kNumberOfBits / 64 + (!!(kNumberOfBits % 64))) * 8];
public:
	__host__ __device__ char* getBytes() {
		return bytes;
	}
};

class Result {
	char bytes[(kNumberOfPairs / 64 + (!!(kNumberOfPairs % 64))) * 8];
public:
	__host__ __device__ char* getBytes() {
		return bytes;
	}
};

void generateInput(Sequence* bits);
void printSequence(Sequence& sequence);
__host__ __device__ unsigned long long* getWord(char* bits, unsigned long long j);
__host__ __device__ unsigned long long* getWord32(char* bits, unsigned long long j);
__host__ __device__ char checkDistance(Sequence& sequence1, Sequence& sequence2, unsigned long long nrOfBits);
__host__ __device__ inline void k2ij(unsigned long long  k, unsigned long long* i, unsigned long long* j);
__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j);
__host__ __device__ inline void SetBit(char* array, unsigned long long index, char value)
{
	array[index / 8] = (array[index / 8] & (~(1 << (index % 8)))) | ((!!value) << (index % 8));
}
__host__ __device__ inline char GetBit(char* array, unsigned long long index)
{
	return array[index / 8] >> (index % 8) & 1;
}
__global__ void hammingGPU(Sequence* d_sequences, char* results, unsigned long long nrOfSeq, unsigned long long nrOfBits,
	unsigned long long offset = 0);
__global__ void hammingGPUPairs(Sequence* d_sequences, char* results, unsigned long long nrOfSeq, unsigned long long nrOfBits,
	unsigned long long offset = 0);

//naprawiæ tooo:
__global__ void hammingGPU(Sequence* d_sequences, char* results, unsigned long long nrOfSeq, unsigned long long nrOfBits,
	unsigned long long offset) {
	unsigned long long threadId = threadIdx.x + blockIdx.x * blockDim.x + offset;
	for (unsigned long long i = 0; i < threadId; i++) {
		//SetBit(results->getBytes(), ij2k(i, threadId), checkDistance(d_sequences[threadId], d_sequences[i], nrOfBits));
		results[ij2k(threadId, i)] = checkDistance(d_sequences[threadId], d_sequences[i], nrOfBits);
	}
}

__global__ void hammingGPUPairs(Sequence* d_sequences, char* results, unsigned long long nrOfSeq, unsigned long long nrOfBits,
	unsigned long long offset) {
	unsigned long long threadId = threadIdx.x + blockIdx.x * blockDim.x + offset;
		//SetBit(results->getBytes(), ij2k(i, threadId), checkDistance(d_sequences[threadId], d_sequences[i], nrOfBits));
	unsigned long long s1, s2;
	k2ij(threadId, &s1, &s2);
	results[threadId] = checkDistance(d_sequences[s1], d_sequences[s2], nrOfBits);
}

std::vector<std::pair<unsigned long long, unsigned long long>> getPairs(char* results, unsigned long long* sum) {
	*sum = 0;
	std::vector<std::pair<unsigned long long, unsigned long long>> pairs;
	for (int i = 0; i < kNumberOfPairs; i++) {
		if (results[i] == 1) {
			unsigned long long s1, s2;
			k2ij(i, &s1, &s2);
			if (s1 != s2) {
				*sum += 1;
				pairs.push_back(std::make_pair(s1, s2));
			}
		}
	}
	return pairs;
}

auto pairsGPU(Sequence* h_sequence) {
	Sequence* d_sequence;
	char *h_results, *d_results;
	h_results = new char[kNumberOfPairs];
	cudaMalloc(&d_sequence, sizeof(Sequence) * kNumberOfSequences);
	cudaMemcpy(d_sequence, h_sequence, sizeof(Sequence) * kNumberOfSequences, cudaMemcpyHostToDevice);
	cudaMalloc(&d_results, sizeof(char) * kNumberOfPairs);
	cudaMemcpy(d_results, h_results, sizeof(char) * kNumberOfPairs, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	//hammingGPUPairs << < kNumberOfPairs / 1024, 1024 >> > (d_sequence, d_results, kNumberOfSequences, kNumberOfBits);
	//if (kNumberOfPairs % 1024) {
	//	hammingGPUPairs << < 1, kNumberOfPairs % 1024 >> > (d_sequence, d_results, kNumberOfSequences, kNumberOfBits,
	//		kNumberOfPairs - kNumberOfPairs % 1024);
	//}
	hammingGPU << < kNumberOfSequences / 1024, 1024 >> > (d_sequence, d_results, kNumberOfSequences, kNumberOfBits);
	if (kNumberOfSequences % 1024) {
		hammingGPU << < 1, kNumberOfSequences % 1024 >> > (d_sequence, d_results, kNumberOfSequences, kNumberOfBits,
			kNumberOfSequences - kNumberOfSequences % 1024);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);

	cudaMemcpy(h_results, d_results, sizeof(char) * kNumberOfPairs, cudaMemcpyDeviceToHost);

	unsigned long long sum = 0;
	const auto& pairs = getPairs(h_results, &sum);
	for (const auto& pair : pairs) {
		//std::cout << "Sequence 1: " << pair.first <<
		//	", sequence 2: " << pair.second << std::endl;
		std::cout << "Sequence 1: ";
		printSequence(h_sequence[pair.first]);
		std::cout << " , sequence 2: ";
		printSequence(h_sequence[pair.second]);
		int ret = checkDistance(h_sequence[pair.first], h_sequence[pair.second], kNumberOfBits);
		std::cout << std::endl;
	}
	std::cout << "sum gpu: " << sum << ", time: " << time << std::endl;
	return pairs;
}

auto pairsCPU(Sequence* sequences) {
	Sequence* d_sequence;
	char *results;
	results = new char[kNumberOfPairs];
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	for (int i = 0; i < kNumberOfSequences; i++) {
		for (int j = 0; j < i; j++) {
			//SetBit(results->getBytes(), ij2k(j, i), checkDistance(sequences[i], sequences[j], kNumberOfBits));
			results[ij2k(i, j)] = checkDistance(sequences[i], sequences[j], kNumberOfBits);
		}
	}
	//for (unsigned long long i = 0; i < kNumberOfPairs; i++) {
	//	unsigned long long s1, s2;
	//	k2ij(i, &s1, &s2);
	//	results[i] = checkDistance(sequences[s1], sequences[s2], kNumberOfBits);
	//}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);

	unsigned long long sum = 0;
	//std::vector<std::pair<unsigned long long, unsigned long long>> pairs;
	//for (int i = 0; i < kNumberOfPairs; i++) {
	//	if ((short int)GetBit(results->getBytes(), i) == 1) {
	//		sum += 1;
	//		unsigned long long s1, s2;
	//		k2ij(i, &s1, &s2);
	//		pairs.push_back(std::make_pair(s1, s2));
	//	}
	//}
	const auto& pairs = getPairs(results, &sum);
	//for (const auto& pair : hmm) {
	//	//std::cout << "Sequence 1: " << pair.first <<
	//	//	", sequence 2: " << pair.second<< std::endl;
	//	std::cout << "Sequence 1: ";
	//	printSequence(sequences[pair.first]);
	//	std::cout << " , sequence 2: ";
	//	printSequence(sequences[pair.second]);
	//	int ret = checkDistance(sequences[pair.first], sequences[pair.second], kNumberOfBits);
	//	std::cout << std::endl;
	//}

	for (const auto& pair : pairs) {
		//std::cout << "Sequence 1: " << pair.first <<
		//	", sequence 2: " << pair.second<< std::endl;
		//std::cout << "Sequence 1: ";
		//printSequence(sequences[pair.first]);
		//std::cout << " , sequence 2: ";
		//printSequence(sequences[pair.second]);
		//int ret = checkDistance(sequences[pair.first], sequences[pair.second], kNumberOfBits);
		//std::cout << std::endl;
	}
	std::cout << "sum cpu: " << sum << ", time: " << time << std::endl;
	return pairs;
}

int main() {
	Sequence* sequences = new Sequence[kNumberOfSequences];
	generateInput(sequences);
	auto pairs1 = pairsCPU(sequences);
	auto pairs2 = pairsGPU(sequences);
	for (const auto& pair : pairs1) {
		if (find(pairs2.begin(), pairs2.end(), pair) == pairs2.end()) {
			std::cout << "pair not found in gpu: " << "s1: " << pair.first << " ";
			printSequence(sequences[pair.first]);
			std::cout << ", " << "s2: " << pair.second << " ";
			printSequence(sequences[pair.second]);
			std::cout << std::endl;
		}
	}
    return 0;
}

void generateInput(Sequence* bits) {
	std::mt19937_64 random;
	int seed = std::random_device()();
	random.seed(seed);
	for (int i = 0; i < kNumberOfSequences; i++) {
		for (int j = 0; j < kNumberOfBits / 64; j++) {
			*getWord(bits[i].getBytes(), j) = random();
		}
		*getWord(bits[i].getBytes(), kNumberOfBits / 64) = random() >> (64 - (kNumberOfBits % 64));
	}
}

void printSequence(Sequence& sequence) {
	for (unsigned long long i = 0; i < kNumberOfBits; i++) {
		std::cout << (sequence.getBytes()[i / 8] >> (i % 8) & 1);
	}
}

__host__ __device__ unsigned long long* getWord(char* bits, unsigned long long j) {
	return (unsigned long long*)(bits + j * 64 / 8);
}

__host__ __device__ unsigned long long* getWord32(char* bits, unsigned long long j) {
	return (unsigned long long*)(bits + j * 32 / 8);
}

__host__ __device__ char checkDistance(Sequence& sequence1, Sequence& sequence2, unsigned long long nrOfBits) {
	int diff = 0;
	for (int j = 0; j < (nrOfBits + 63) / 64; ++j) {
		unsigned long long int a, b, xor;
		a = *(getWord(sequence1.getBytes(), j));
		b = *(getWord(sequence2.getBytes(), j));
		xor = a ^ b;
		diff += xor == 0 ? 0 : (xor & (xor -1) ? 2 : 1);
		if (diff > 1) {
			return 0;
		}
	}
	return !!diff;
}

__host__ __device__ inline void k2ij(unsigned long long  k, unsigned long long* i, unsigned long long* j) {
	*i = (unsigned int)ceilf((0.5f * (-1 + sqrtf(1 + 8 * (k + 1)))));
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j) {
	return i * (i - 1) / 2 + j;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iterator>
#include <set>

constexpr unsigned long long kNumberOfBits = 1000;
constexpr unsigned long long kNumberOfSequences = 10000;
constexpr unsigned long long kNumberOfPairs = (kNumberOfSequences * (kNumberOfSequences - 1)) / 2;
constexpr double kSequencePercent = 0.3;
constexpr bool comparePairs = true;
constexpr bool printSequences = false;

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
std::vector<std::pair<unsigned long long, unsigned long long>> getPairs(char* results, unsigned long long* sum);
__host__ __device__ unsigned long long* getWord(char* bits, unsigned long long j);
__host__ __device__ char checkDistance(Sequence& sequence1, Sequence& sequence2, unsigned long long nrOfBits);
__host__ __device__ inline void k2ij(unsigned long long  k, unsigned long long* i, unsigned long long* j);
__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j);
__global__ void hammingGPU(Sequence* d_sequences, char* results, unsigned long long nrOfBits,
	unsigned long long offset = 0);
__global__ void hammingGPUPairs(Sequence* d_sequences, char* results, unsigned long long nrOfBits,
	unsigned long long offset = 0);
__host__ __device__ inline void setBit(char* array, unsigned long long index, char value);
__host__ __device__ inline char getBit(char* array, unsigned long long index);

__global__ void hammingGPU(Sequence* d_sequences, char* results, unsigned long long nrOfBits,
	unsigned long long offset) {
	unsigned long long threadId = threadIdx.x + blockIdx.x * blockDim.x + offset;
	for (unsigned long long i = 0; i < threadId; i++) {
		results[ij2k(threadId, i)] = checkDistance(d_sequences[threadId], d_sequences[i], nrOfBits);
	}
}

__global__ void hammingGPUPairs(Sequence* d_sequences, char* results, unsigned long long nrOfBits,
	unsigned long long offset) {
	unsigned long long threadId = threadIdx.x + blockIdx.x * blockDim.x + offset;
	unsigned long long s1, s2;
	k2ij(threadId, &s1, &s2);
	results[threadId] = checkDistance(d_sequences[s1], d_sequences[s2], nrOfBits);
}

auto pairsGPU(Sequence* h_sequence) {
	std::cout << std::endl << "Finding pairs on GPU.." << std::endl;
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

	if (comparePairs) {
		hammingGPUPairs << < kNumberOfPairs / 1024, 1024 >> > (d_sequence, d_results, kNumberOfBits);
		if (kNumberOfPairs % 1024) {
			hammingGPUPairs << < 1, kNumberOfPairs % 1024 >> > (d_sequence, d_results, kNumberOfBits,
				kNumberOfPairs - kNumberOfPairs % 1024);
		}
	}
	else {
		hammingGPU << < kNumberOfSequences / 1024, 1024 >> > (d_sequence, d_results, kNumberOfBits);
		if (kNumberOfSequences % 1024) {
			hammingGPU << < 1, kNumberOfSequences % 1024 >> > (d_sequence, d_results, kNumberOfBits,
				kNumberOfSequences - kNumberOfSequences % 1024);
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(h_results, d_results, sizeof(char) * kNumberOfPairs, cudaMemcpyDeviceToHost);

	unsigned long long sum = 0;
	const auto& pairs = getPairs(h_results, &sum);
	std::cout << "Pairs found on GPU: " << sum << ", time elapsed: " <<
		time << "ms" << std::endl;
	return pairs;
}

auto pairsCPU(Sequence* sequences) {
	std::cout << std::endl << "Finding pairs on CPU.." << std::endl;
	char *results;
	results = new char[kNumberOfPairs];
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	if (comparePairs) {
		for (unsigned long long i = 0; i < kNumberOfPairs; i++) {
			unsigned long long s1, s2;
			k2ij(i, &s1, &s2);
			results[i] = checkDistance(sequences[s1], sequences[s2], kNumberOfBits);
		}
	}
	else {
		for (int i = 0; i < kNumberOfSequences; i++) {
			for (int j = 0; j < i; j++) {
				results[ij2k(i, j)] = checkDistance(sequences[i], sequences[j], kNumberOfBits);
			}
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);

	unsigned long long sum = 0;
	const auto& pairs = getPairs(results, &sum);
	std::cout << "Pairs found on CPU: " << sum << ", time elapsed: " <<
		time << "ms" << std::endl;
	return pairs;
}

int main() {
	std::cout << "Finding pairs of " << kNumberOfBits << "-bit sequences with " <<
		"Hamming distance equal to 1" << std::endl;
	std::cout << "Number of sequences: " << kNumberOfSequences << std::endl;
	std::cout << "Using algorithm: ";
	if (comparePairs) {
		std::cout << "comparing each pair";
	}
	else {
		std::cout << "comparing each sequence iteratively";
	}
	std::cout << std::endl;
	Sequence* sequences = new Sequence[kNumberOfSequences];
	generateInput(sequences);
	auto pairs1 = pairsCPU(sequences);
	auto pairs2 = pairsGPU(sequences);
	//pairs2.push_back(std::make_pair(25ull, 56ull));
	if (pairs1 != pairs2) {
		std::cout << std::endl << "Returned pairs differ" << std::endl;
		std::vector<std::pair<unsigned long long, unsigned long long>> diff;
		std::set<std::pair<unsigned long long, unsigned long long>> set1(pairs1.begin(), pairs1.end()),
			set2(pairs2.begin(), pairs2.end());
		std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(),
			std::back_inserter(diff));
		std::set_difference(set2.begin(), set2.end(), set1.begin(), set1.end(),
			std::back_inserter(diff));
		std::cout << "Mismatched pairs: " << std::endl;
		for (const auto& pair : diff) {
			std::cout << "Sequence 1: [" << pair.first << "] ";
			printSequence(sequences[pair.first]);
			std::cout << ", sequence 2: [" << pair.second << "] ";
			printSequence(sequences[pair.second]);
			std::cout << std::endl;
		}
	}
	else {
		std::cout << std::endl << "Returned pairs are the same" << std::endl;
		if (printSequences) {
			for (const auto& pair: pairs1) {
				std::cout << "Sequence 1: [" << pair.first << "] ";
				printSequence(sequences[pair.first]);
				std::cout << ", sequence 2: [" << pair.second << "] ";
				printSequence(sequences[pair.second]);
				std::cout << std::endl;
			}
		}
	}
    return 0;
}

std::vector<std::pair<unsigned long long, unsigned long long>> getPairs(char* results, unsigned long long* sum) {
	*sum = 0;
	std::vector<std::pair<unsigned long long, unsigned long long>> pairs;
	for (int i = 0; i < kNumberOfPairs; i++) {
		if (results[i] == 1) {
			unsigned long long s1, s2;
			k2ij(i, &s1, &s2);
			if (s1 > s2) {
				*sum += 1;
				pairs.push_back(std::make_pair(s1, s2));
			}
		}
	}
	return pairs;
}

void generateInput(Sequence* bits) {
	std::cout << std::endl << "Generating sequences.." << std::endl;
	std::mt19937_64 random;
	int seed = std::random_device()();
	random.seed(seed);
	for (int i = 0; i < kNumberOfSequences; i++) {
		for (int j = 0; j < kNumberOfBits / 64; j++) {
			*getWord(bits[i].getBytes(), j) = random();
		}
		*getWord(bits[i].getBytes(), kNumberOfBits / 64) = random() >> (64 - (kNumberOfBits % 64));
	}
	std::uniform_int_distribution<unsigned long long> distribution(0, kNumberOfSequences - 1);
	for (int i = 0; i < kSequencePercent * kNumberOfSequences; i++) {
		unsigned long long s1, s2;
		s1 = distribution(random);
		s2 = distribution(random);
		for (int j = 0; j < kNumberOfBits / 64; j++) {
			*getWord(bits[s2].getBytes(), j) = *getWord(bits[s1].getBytes(), j);
		}
		*getWord(bits[s2].getBytes(), kNumberOfBits / 64) = 
			*getWord(bits[s1].getBytes(), kNumberOfBits / 64);
		char bit = getBit(bits[s1].getBytes(), kNumberOfBits - 1) == 1 ? 0 : 1;
		setBit(bits[s2].getBytes(), kNumberOfBits - 1, bit);
	}
}

void printSequence(Sequence& sequence) {
	for (unsigned long long i = 0; i < kNumberOfBits; i++) {
		std::cout << (sequence.getBytes()[i / 8] >> (i % 8) & 1);
	}
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

__host__ __device__ unsigned long long* getWord(char* bits, unsigned long long j) {
	return (unsigned long long*)(bits + j * 64 / 8);
}

__host__ __device__ inline void k2ij(unsigned long long  k, unsigned long long* i, unsigned long long* j) {
	*i = (unsigned int)ceilf((0.5f * (-1 + sqrtf(1 + 8 * (k + 1)))));
	*j = (unsigned int)(((k + 1) - 0.5 * (*i) * ((*i) - 1)) - 1);
}

__host__ __device__ unsigned long long ij2k(unsigned long long i, unsigned long long j) {
	return i * (i - 1) / 2 + j;
}

__host__ __device__ inline void setBit(char* array, unsigned long long index, char value)
{
	array[index / 8] = (array[index / 8] & (~(1 << (index % 8)))) | ((!!value) << (index % 8));
}

__host__ __device__ inline char getBit(char* array, unsigned long long index)
{
	return array[index / 8] >> (index % 8) & 1;
}
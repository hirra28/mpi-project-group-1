// generate.cpp
#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char* argv[]) {

    size_t N = 10000000; // default = 10M (Medium)
    if (argc > 1) {
        N = std::stoull(argv[1]);
    }

    std::cout << "Generating " << N << " random records...\n";

    // Fixed seed for reproducibility
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 10000.0);

    std::ofstream bin("C:\\Users\\DELL\\mpi-project-group-1\\data.bin", std::ios::binary);

    if (!bin.is_open()) {
        std::cout << "ERROR: Cannot open output file!\n";
        return 1;
    }

    // Write N
    bin.write(reinterpret_cast<char*>(&N), sizeof(size_t));

    // Generate and write trip_distance (N values)
    for (size_t i = 0; i < N; i++) {
        double val = dist(rng);
        bin.write(reinterpret_cast<char*>(&val), sizeof(double));
    }

    // Generate and write fare_amount (N values)
    for (size_t i = 0; i < N; i++) {
        double val = dist(rng);
        bin.write(reinterpret_cast<char*>(&val), sizeof(double));
    }

    std::cout << "Done! Saved " << N << " records to data.bin\n";
    return 0;
}
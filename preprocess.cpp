#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {

    // Default limit = 1000000 (Small), pass argument to change
    size_t LIMIT = 1000000;
    if (argc > 1) {
        LIMIT = std::stoull(argv[1]);
    }

    std::ifstream csv("C:\\Users\\Husna\\mpi-project-group-1\\data\\yellow_tripdata_2015-01.csv");
    std::ofstream bin("C:\\Users\\Husna\\mpi-project-group-1\\data.bin", std::ios::binary);

    if (!csv.is_open()) {
        std::cout << "ERROR: Cannot open CSV file. Check the path!\n";
        return 1;
    }

    std::string line;
    std::getline(csv, line); // skip header row

    std::vector<double> trip_distance, fare_amount;

    while (std::getline(csv, line)) {
        if (trip_distance.size() >= LIMIT) break; // STOP at limit

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> cols;

        while (std::getline(ss, token, ','))
            cols.push_back(token);

        try {
            double dist = std::stod(cols[4]); // trip_distance
            double fare = std::stod(cols[5]); // fare_amount
            trip_distance.push_back(dist);
            fare_amount.push_back(fare);
        } catch (...) {
            continue; // skip bad rows
        }
    }

    size_t N = trip_distance.size();
    bin.write(reinterpret_cast<char*>(&N), sizeof(size_t));
    bin.write(reinterpret_cast<char*>(trip_distance.data()), N * sizeof(double));
    bin.write(reinterpret_cast<char*>(fare_amount.data()), N * sizeof(double));

    std::cout << "Done! Saved " << N << " records to data.bin\n";
    return 0;
}
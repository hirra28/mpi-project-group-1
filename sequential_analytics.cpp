// sequential_analytics.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

// ─────────────────────────────────────────
// LOAD DATA FROM data.bin
// ─────────────────────────────────────────
void loadData(const std::string& path,
              std::vector<double>& trip_distance,
              std::vector<double>& fare_amount) {

    std::ifstream bin(path, std::ios::binary);
    if (!bin.is_open()) {
        std::cout << "ERROR: Cannot open data.bin!\n";
        exit(1);
    }

    size_t N;
    bin.read(reinterpret_cast<char*>(&N), sizeof(size_t));

    trip_distance.resize(N);
    fare_amount.resize(N);

    bin.read(reinterpret_cast<char*>(trip_distance.data()), N * sizeof(double));
    bin.read(reinterpret_cast<char*>(fare_amount.data()),   N * sizeof(double));

    std::cout << "Loaded " << N << " records from data.bin\n\n";
}

// ─────────────────────────────────────────
// TASK 1 — BASIC STATISTICS
// ─────────────────────────────────────────
void task1_basicStats(const std::vector<double>& data, std::ofstream& csv) {
    auto start = std::chrono::high_resolution_clock::now();

    double sum = 0, minVal = data[0], maxVal = data[0];
    for (double v : data) {
        sum += v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    double mean = sum / data.size();

    double variance = 0;
    for (double v : data)
        variance += (v - mean) * (v - mean);
    variance /= data.size();
    double stddev = std::sqrt(variance);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Task 1 - Basic Statistics:\n";
    std::cout << "  Mean     : " << mean    << "\n";
    std::cout << "  Variance : " << variance << "\n";
    std::cout << "  Std Dev  : " << stddev  << "\n";
    std::cout << "  Min      : " << minVal  << "\n";
    std::cout << "  Max      : " << maxVal  << "\n";
    std::cout << "  Time     : " << ms << " ms\n\n";

    csv << "BasicStats," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 2 — HISTOGRAM
// ─────────────────────────────────────────
void task2_histogram(const std::vector<double>& data, std::ofstream& csv) {
    auto start = std::chrono::high_resolution_clock::now();

    int bins = 10;
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    double binWidth = (maxVal - minVal) / bins;

    std::vector<int> histogram(bins, 0);
    for (double v : data) {
        int idx = (int)((v - minVal) / binWidth);
        if (idx >= bins) idx = bins - 1;
        histogram[idx]++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Task 2 - Histogram (10 bins):\n";
    for (int i = 0; i < bins; i++)
        std::cout << "  Bin " << i+1 << ": " << histogram[i] << " records\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    csv << "Histogram," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 3 — SORTING
// ─────────────────────────────────────────
void task3_sorting(std::vector<double> data, std::ofstream& csv) {
    auto start = std::chrono::high_resolution_clock::now();

    std::sort(data.begin(), data.end());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Task 3 - Sorting:\n";
    std::cout << "  First value : " << data.front() << "\n";
    std::cout << "  Last value  : " << data.back()  << "\n";
    std::cout << "  Time        : " << ms << " ms\n\n";

    csv << "Sorting," << data.size() << "," << ms << "\n";
}
// ─────────────────────────────────────────
// TASK 4 — PEARSON CORRELATION
// ─────────────────────────────────────────
void task4_correlation(const std::vector<double>& x,
                       const std::vector<double>& y,
                       std::ofstream& csv) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t N = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (size_t i = 0; i < N; i++) {
        sumX  += x[i];
        sumY  += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }

    double correlation = (N * sumXY - sumX * sumY) /
        std::sqrt((N * sumX2 - sumX * sumX) * (N * sumY2 - sumY * sumY));

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Task 4 - Pearson Correlation:\n";
    std::cout << "  Correlation (distance vs fare): " << correlation << "\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    csv << "Correlation," << N << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 5 — MOVING AVERAGE
// ─────────────────────────────────────────
void task5_movingAverage(const std::vector<double>& data, std::ofstream& csv) {
    auto start = std::chrono::high_resolution_clock::now();

    int window = 100;
    std::vector<double> movAvg(data.size() - window + 1);
    double windowSum = 0;

    for (int i = 0; i < window; i++)
        windowSum += data[i];
    movAvg[0] = windowSum / window;

    for (size_t i = 1; i < movAvg.size(); i++) {
        windowSum += data[i + window - 1];
        windowSum -= data[i - 1];
        movAvg[i] = windowSum / window;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Task 5 - Moving Average (window=100):\n";
    std::cout << "  First moving avg : " << movAvg.front() << "\n";
    std::cout << "  Last moving avg  : " << movAvg.back()  << "\n";
    std::cout << "  Time             : " << ms << " ms\n\n";

    csv << "MovingAverage," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 6 — OUTLIER DETECTION (Z-SCORE)
// ─────────────────────────────────────────
void task6_outliers(const std::vector<double>& data, std::ofstream& csv) {
    auto start = std::chrono::high_resolution_clock::now();

    double sum = 0;
    for (double v : data) sum += v;
    double mean = sum / data.size();

    double variance = 0;
    for (double v : data)
        variance += (v - mean) * (v - mean);
    double stddev = std::sqrt(variance / data.size());

    int outlierCount = 0;
    for (double v : data) {
        double zscore = std::abs((v - mean) / stddev);
        if (zscore > 3.0) outlierCount++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Task 6 - Outlier Detection (Z-score > 3):\n";
    std::cout << "  Outliers found : " << outlierCount << "\n";
    std::cout << "  Time           : " << ms << " ms\n\n";

    csv << "OutlierDetection," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────
int main() {
    std::vector<double> trip_distance, fare_amount;

    // Load data
    loadData("C:\\Users\\Husna\\mpi-project-group-1\\data.bin",
             trip_distance, fare_amount);

    // Open CSV results file
    std::ofstream csv("C:\\Users\\Husna\\mpi-project-group-1\\sequential_results.csv");
    csv << "Task,DataSize,TimeMs\n";

    // Run all 6 tasks
    task1_basicStats   (trip_distance,              csv);
    task2_histogram    (trip_distance,              csv);
    task3_sorting      (trip_distance,              csv);
    task4_correlation  (trip_distance, fare_amount, csv);
    task5_movingAverage(trip_distance,              csv);
    task6_outliers     (trip_distance,              csv);

    csv.close();
    std::cout << "All tasks done! Results saved to sequential_results.csv\n";
    return 0;
}
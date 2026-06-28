// analytics.cpp  —  sequential version (no MPI)
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ─────────────────────────────────────────
// LOAD DATA
// ─────────────────────────────────────────
void loadData(const std::string& path,
              std::vector<double>& trip_distance,
              std::vector<double>& fare_amount)
{
    std::ifstream bin(path, std::ios::binary);
    if (!bin.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << "\n";
        std::exit(1);
    }

    size_t N = 0;
    bin.read(reinterpret_cast<char*>(&N), sizeof(size_t));

    trip_distance.resize(N);
    fare_amount.resize(N);

    bin.read(reinterpret_cast<char*>(trip_distance.data()), N * sizeof(double));
    bin.read(reinterpret_cast<char*>(fare_amount.data()),   N * sizeof(double));

    std::cout << "Loaded " << N << " records from " << path << "\n\n";
}

// ─────────────────────────────────────────
// TASK 1 — BASIC STATISTICS
// ─────────────────────────────────────────
void task1_basicStats(const std::vector<double>& data,
                      std::ofstream& csv)
{
    auto t0 = Clock::now();

    double sum = 0, mn = data[0], mx = data[0];
    for (double v : data) {
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double mean = sum / data.size();

    double var = 0;
    for (double v : data)
        var += (v - mean) * (v - mean);
    var /= data.size();

    double ms = Ms(Clock::now() - t0).count();

    std::cout << "Task 1 - Basic Statistics:\n";
    std::cout << "  Mean     : " << mean           << "\n";
    std::cout << "  Variance : " << var            << "\n";
    std::cout << "  Std Dev  : " << std::sqrt(var) << "\n";
    std::cout << "  Min      : " << mn             << "\n";
    std::cout << "  Max      : " << mx             << "\n";
    std::cout << "  Time     : " << ms             << " ms\n\n";

    csv << "BasicStats," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 2 — HISTOGRAM
// ─────────────────────────────────────────
void task2_histogram(const std::vector<double>& data,
                     double globalMin, double globalMax,
                     std::ofstream& csv)
{
    auto t0 = Clock::now();

    const int bins = 10;
    double binWidth = (globalMax - globalMin) / bins;
    std::vector<int> hist(bins, 0);

    for (double v : data) {
        int idx = static_cast<int>((v - globalMin) / binWidth);
        if (idx >= bins) idx = bins - 1;
        hist[idx]++;
    }

    double ms = Ms(Clock::now() - t0).count();

    std::cout << "Task 2 - Histogram (10 bins):\n";
    for (int i = 0; i < bins; i++)
        std::cout << "  Bin " << i + 1 << ": " << hist[i] << " records\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    csv << "Histogram," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 3 — SORTING
// ─────────────────────────────────────────
void task3_sorting(std::vector<double> data,   // copy intentional
                   std::ofstream& csv)
{
    auto t0 = Clock::now();
    std::sort(data.begin(), data.end());
    double ms = Ms(Clock::now() - t0).count();

    std::cout << "Task 3 - Sorting:\n";
    std::cout << "  First value : " << data.front() << "\n";
    std::cout << "  Last value  : " << data.back()  << "\n";
    std::cout << "  Time        : " << ms            << " ms\n\n";

    csv << "Sorting," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 4 — PEARSON CORRELATION
// ─────────────────────────────────────────
void task4_correlation(const std::vector<double>& X,
                       const std::vector<double>& Y,
                       std::ofstream& csv)
{
    auto t0 = Clock::now();

    double sumX=0, sumY=0, sumXY=0, sumX2=0, sumY2=0;
    size_t N = X.size();
    for (size_t i = 0; i < N; i++) {
        sumX  += X[i];
        sumY  += Y[i];
        sumXY += X[i] * Y[i];
        sumX2 += X[i] * X[i];
        sumY2 += Y[i] * Y[i];
    }

    double n = static_cast<double>(N);
    double corr = (n * sumXY - sumX * sumY) /
        std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    double ms = Ms(Clock::now() - t0).count();

    std::cout << "Task 4 - Pearson Correlation:\n";
    std::cout << "  Correlation (distance vs fare): " << corr << "\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    csv << "Correlation," << N << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 5 — MOVING AVERAGE
// ─────────────────────────────────────────
void task5_movingAverage(const std::vector<double>& data,
                         std::ofstream& csv)
{
    auto t0 = Clock::now();

    const int window = 100;
    std::vector<double> avg;

    if (static_cast<int>(data.size()) >= window) {
        avg.resize(data.size() - window + 1);
        double wsum = 0;
        for (int i = 0; i < window; i++) wsum += data[i];
        avg[0] = wsum / window;
        for (size_t i = 1; i < avg.size(); i++) {
            wsum += data[i + window - 1];
            wsum -= data[i - 1];
            avg[i] = wsum / window;
        }
    }

    double ms = Ms(Clock::now() - t0).count();

    std::cout << "Task 5 - Moving Average (window=100):\n";
    if (!avg.empty()) {
        std::cout << "  First moving avg : " << avg.front() << "\n";
        std::cout << "  Last moving avg  : " << avg.back()  << "\n";
    }
    std::cout << "  Time             : " << ms << " ms\n\n";

    csv << "MovingAverage," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// TASK 6 — OUTLIER DETECTION (Z-SCORE)
// ─────────────────────────────────────────
void task6_outliers(const std::vector<double>& data,
                    double mean, double stddev,
                    std::ofstream& csv)
{
    auto t0 = Clock::now();

    int count = 0;
    for (double v : data) {
        if (std::abs((v - mean) / stddev) > 3.0)
            count++;
    }

    double ms = Ms(Clock::now() - t0).count();

    std::cout << "Task 6 - Outlier Detection (Z-score > 3):\n";
    std::cout << "  Outliers found : " << count << "\n";
    std::cout << "  Time           : " << ms    << " ms\n\n";

    csv << "OutlierDetection," << data.size() << "," << ms << "\n";
}

// ─────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────
int main()
{
    std::vector<double> trip_distance, fare_amount;
    loadData("C:\\Users\\HP\\mpi-project-group-1\\data.bin",
             trip_distance, fare_amount);

    size_t N = trip_distance.size();

    // Pre-compute global min, max, mean, stddev
    double gMin = *std::min_element(trip_distance.begin(), trip_distance.end());
    double gMax = *std::max_element(trip_distance.begin(), trip_distance.end());

    double gSum = 0;
    for (double v : trip_distance) gSum += v;
    double gMean = gSum / N;

    double gVar = 0;
    for (double v : trip_distance) gVar += (v - gMean) * (v - gMean);
    gVar /= N;
    double gStddev = std::sqrt(gVar);

    std::ofstream csv("C:\\Users\\HP\\mpi-project-group-1\\sequential_results.csv");
    csv << "Task,DataSize,TimeMs\n";

    std::cout << "Running sequential analytics on " << N << " records\n\n";

    task1_basicStats   (trip_distance,                   csv);
    task2_histogram    (trip_distance, gMin, gMax,       csv);
    task3_sorting      (trip_distance,                   csv);
    task4_correlation  (trip_distance, fare_amount,      csv);
    task5_movingAverage(trip_distance,                   csv);
    task6_outliers     (trip_distance, gMean, gStddev,   csv);

    csv.close();
    std::cout << "All tasks done! Results saved to sequential_results.csv\n";
    return 0;
}
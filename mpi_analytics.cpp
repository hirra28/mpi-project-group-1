// mpi_analytics.cpp — MPI parallel version
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

const std::string BASE_DIR = "C:\\Users\\Dell\\mpi-project-group-1\\";

// ─────────────────────────────────────────
// LOAD FULL DATA (only Rank 0 does this)
// ─────────────────────────────────────────
void loadData(const std::string& path,
              std::vector<double>& trip_distance,
              std::vector<double>& fare_amount,
              size_t& N)
{
    std::ifstream bin(path, std::ios::binary);
    if (!bin.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    bin.read(reinterpret_cast<char*>(&N), sizeof(size_t));
    trip_distance.resize(N);
    fare_amount.resize(N);

    bin.read(reinterpret_cast<char*>(trip_distance.data()), N * sizeof(double));
    bin.read(reinterpret_cast<char*>(fare_amount.data()),   N * sizeof(double));
}

// ─────────────────────────────────────────
// HELPER — compute send counts/displacements for Scatterv
// ─────────────────────────────────────────
void computeCounts(size_t N, int worldSize,
                   std::vector<int>& counts,
                   std::vector<int>& displs)
{
    counts.resize(worldSize);
    displs.resize(worldSize);

    int base = N / worldSize;
    int rem  = N % worldSize;
    int offset = 0;

    for (int i = 0; i < worldSize; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
}

// ─────────────────────────────────────────
// LOG RESULT — only Rank 0 writes
// ─────────────────────────────────────────
void logResult(std::ofstream& csv, const std::string& task,
               size_t dataSize, int nodes, double ms)
{
    csv << task << "," << dataSize << "," << nodes << "," << ms << "\n";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // ── Dataset size override via command line (optional) ──
    size_t requestedN = 0;
    if (argc > 1) requestedN = std::stoull(argv[1]);

    std::vector<double> trip_distance, fare_amount;
    size_t N = 0;

    // ── STEP 1: Rank 0 loads full data ──
    if (rank == 0) {
        loadData(BASE_DIR + "data.bin", trip_distance, fare_amount, N);
        if (requestedN > 0 && requestedN < N) {
            N = requestedN;
            trip_distance.resize(N);
            fare_amount.resize(N);
        }
        std::cout << "Rank 0: Loaded " << N << " records, running on "
                   << worldSize << " nodes\n\n";
    }

    // ── STEP 2: Broadcast N to all ranks ──
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // ── STEP 3: Compute scatter counts/displacements ──
    std::vector<int> counts, displs;
    computeCounts(N, worldSize, counts, displs);

    int localN = counts[rank];
    std::vector<double> localDist(localN), localFare(localN);

    // ── STEP 4: Scatter trip_distance and fare_amount ──
    MPI_Scatterv(trip_distance.data(), counts.data(), displs.data(), MPI_DOUBLE,
                localDist.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(fare_amount.data(), counts.data(), displs.data(), MPI_DOUBLE,
                localFare.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Open results CSV (rank 0 only) ──
    std::ofstream csv;
    if (rank == 0) {
        csv.open(BASE_DIR + "mpi_results.csv", std::ios::app);
        // Write header only if file is empty
        std::ifstream check(BASE_DIR + "mpi_results.csv");
        check.seekg(0, std::ios::end);
        if (check.tellg() == 0) {
            csv << "Task,DataSize,Nodes,TimeMs\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ═══════════════════════════════════════════════
    // TASK 1 — BASIC STATISTICS
    // ═══════════════════════════════════════════════
    {
        double t0 = MPI_Wtime();

        double localSum = 0, localMin = localDist[0], localMax = localDist[0];
        for (double v : localDist) {
            localSum += v;
            if (v < localMin) localMin = v;
            if (v > localMax) localMax = v;
        }

        double globalSum, globalMin, globalMax;
        MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double globalMean = 0;
        if (rank == 0) globalMean = globalSum / N;
        MPI_Bcast(&globalMean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double localVar = 0;
        for (double v : localDist)
            localVar += (v - globalMean) * (v - globalMean);

        double globalVar;
        MPI_Reduce(&localVar, &globalVar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1000.0;

        if (rank == 0) {
            globalVar /= N;
            std::cout << "Task 1 - Basic Statistics:\n";
            std::cout << "  Mean: " << globalMean << "  StdDev: " << std::sqrt(globalVar)
                      << "  Min: " << globalMin << "  Max: " << globalMax
                      << "  Time: " << ms << " ms\n\n";
            logResult(csv, "BasicStats", N, worldSize, ms);
        }
    }

    // ═══════════════════════════════════════════════
    // TASK 2 — HISTOGRAM
    // ═══════════════════════════════════════════════
    {
        double t0 = MPI_Wtime();
        const int bins = 10;

        double localMin = *std::min_element(localDist.begin(), localDist.end());
        double localMax = *std::max_element(localDist.begin(), localDist.end());
        double globalMin, globalMax;
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double binWidth = (globalMax - globalMin) / bins;
        std::vector<int> localHist(bins, 0);

        for (double v : localDist) {
            int idx = static_cast<int>((v - globalMin) / binWidth);
            if (idx >= bins) idx = bins - 1;
            if (idx < 0) idx = 0;
            localHist[idx]++;
        }

        std::vector<int> globalHist(bins, 0);
        MPI_Reduce(localHist.data(), globalHist.data(), bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1000.0;

        if (rank == 0) {
            std::cout << "Task 2 - Histogram:\n";
            for (int i = 0; i < bins; i++)
                std::cout << "  Bin " << i+1 << ": " << globalHist[i] << "\n";
            std::cout << "  Time: " << ms << " ms\n\n";
            logResult(csv, "Histogram", N, worldSize, ms);
        }
    }

    // ═══════════════════════════════════════════════
    // TASK 3 — SORTING (local sort + gather + merge on rank 0)
    // ═══════════════════════════════════════════════
    {
        double t0 = MPI_Wtime();

        std::vector<double> sortedLocal = localDist;
        std::sort(sortedLocal.begin(), sortedLocal.end());

        std::vector<double> gathered;
        if (rank == 0) gathered.resize(N);

        MPI_Gatherv(sortedLocal.data(), localN, MPI_DOUBLE,
                   gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::sort(gathered.begin(), gathered.end()); // final merge step
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1000.0;

        if (rank == 0) {
            std::cout << "Task 3 - Sorting:\n";
            std::cout << "  First: " << gathered.front() << "  Last: " << gathered.back()
                      << "  Time: " << ms << " ms\n\n";
            logResult(csv, "Sorting", N, worldSize, ms);
        }
    }

    // ═══════════════════════════════════════════════
    // TASK 4 — PEARSON CORRELATION
    // ═══════════════════════════════════════════════
    {
        double t0 = MPI_Wtime();

        double sumX=0, sumY=0, sumXY=0, sumX2=0, sumY2=0;
        for (int i = 0; i < localN; i++) {
            sumX  += localDist[i];
            sumY  += localFare[i];
            sumXY += localDist[i] * localFare[i];
            sumX2 += localDist[i] * localDist[i];
            sumY2 += localFare[i] * localFare[i];
        }

        double gSumX, gSumY, gSumXY, gSumX2, gSumY2;
        MPI_Reduce(&sumX,  &gSumX,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sumY,  &gSumY,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sumXY, &gSumXY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sumX2, &gSumX2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sumY2, &gSumY2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1000.0;

        if (rank == 0) {
            double n = static_cast<double>(N);
            double corr = (n * gSumXY - gSumX * gSumY) /
                std::sqrt((n * gSumX2 - gSumX * gSumX) * (n * gSumY2 - gSumY * gSumY));
            std::cout << "Task 4 - Pearson Correlation: " << corr
                      << "  Time: " << ms << " ms\n\n";
            logResult(csv, "Correlation", N, worldSize, ms);
        }
    }

    // ═══════════════════════════════════════════════
    // TASK 5 — MOVING AVERAGE (computed locally per chunk)
    // ═══════════════════════════════════════════════
    {
        double t0 = MPI_Wtime();
        const int window = 100;

        std::vector<double> localAvg;
        if (localN >= window) {
            localAvg.resize(localN - window + 1);
            double wsum = 0;
            for (int i = 0; i < window; i++) wsum += localDist[i];
            localAvg[0] = wsum / window;
            for (int i = 1; i < (int)localAvg.size(); i++) {
                wsum += localDist[i + window - 1];
                wsum -= localDist[i - 1];
                localAvg[i] = wsum / window;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1000.0;

        if (rank == 0) {
            std::cout << "Task 5 - Moving Average (per-node, window=100): Time: "
                      << ms << " ms\n\n";
            logResult(csv, "MovingAverage", N, worldSize, ms);
        }
    }

    // ═══════════════════════════════════════════════
    // TASK 6 — OUTLIER DETECTION (Z-SCORE)
    // ═══════════════════════════════════════════════
    {
        double t0 = MPI_Wtime();

        // Reuse global mean/stddev computed in Task 1 style
        double localSum = 0;
        for (double v : localDist) localSum += v;
        double globalSum;
        MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double globalMean = globalSum / N;

        double localVar = 0;
        for (double v : localDist) localVar += (v - globalMean) * (v - globalMean);
        double globalVar;
        MPI_Allreduce(&localVar, &globalVar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double globalStddev = std::sqrt(globalVar / N);

        int localOutliers = 0;
        for (double v : localDist) {
            if (std::abs((v - globalMean) / globalStddev) > 3.0)
                localOutliers++;
        }

        int globalOutliers;
        MPI_Reduce(&localOutliers, &globalOutliers, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1000.0;

        if (rank == 0) {
            std::cout << "Task 6 - Outliers found: " << globalOutliers
                      << "  Time: " << ms << " ms\n\n";
            logResult(csv, "OutlierDetection", N, worldSize, ms);
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "All tasks done! Results appended to mpi_results.csv\n";
    }

    MPI_Finalize();
    return 0;
}
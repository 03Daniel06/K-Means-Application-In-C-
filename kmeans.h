#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <QWidget>

struct DataPoint {
    std::vector<double> coordinates;
};

class KMeans {
public:
    KMeans();

    void setParameters(int clusters, int pointsPerCluster, int dimensions, double stdDev, int initializations);
    void run();

    double getGroundTruthError() const;
    double getSimilarityMeasure() const;
    double getBestError() const;

    void plotClusters(QWidget *plotWidget);

private:
    void generateData();
    void computeGroundTruthError();
    void computeFriendMatrix(const std::vector<int>& labels, std::vector<std::vector<int>>& friendMatrix);
    void kMeansAlgorithm();

    int numClusters;
    int numPointsPerCluster;
    int numDimensions;
    double standardDeviation;
    int numInitializations;

    std::vector<DataPoint> trainData;
    std::vector<DataPoint> testData;
    std::vector<DataPoint> startPoints;

    std::vector<int> groundTruthLabels;
    std::vector<int> bestLabels;
    std::vector<DataPoint> bestCentroids;

    double groundTruthError;
    double similarityMeasure;
    double bestError;
};

#endif // KMEANS_H

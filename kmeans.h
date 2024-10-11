#ifndef KMEANS_H
#define KMEANS_H

#include <Eigen/Dense>
#include <QWidget>

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
    void computeFriendMatrix(const Eigen::VectorXi &labels, Eigen::MatrixXi &friendMatrix);
    void kMeansAlgorithm();

    int numClusters;
    int numPointsPerCluster;
    int numDimensions;
    double standardDeviation;
    int numInitializations;

    Eigen::MatrixXd trainData;
    Eigen::MatrixXd testData;
    Eigen::MatrixXd startPoints;

    Eigen::VectorXi groundTruthLabels;
    Eigen::VectorXi bestLabels;
    Eigen::MatrixXd bestCentroids;

    double groundTruthError;
    double similarityMeasure;
    double bestError;
};

#endif // KMEANS_H

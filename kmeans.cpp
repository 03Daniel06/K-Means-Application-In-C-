#include "kmeans.h"
#include <QRandomGenerator>
#include <QDebug>
#include <QtCharts>

KMeans::KMeans() {
    numClusters = 3;
    numPointsPerCluster = 50;
    numDimensions = 2;
    standardDeviation = 0.1;
    numInitializations = 10;
}

void KMeans::setParameters(int clusters, int pointsPerCluster, int dimensions, double stdDev, int initializations) {
    numClusters = clusters;
    numPointsPerCluster = pointsPerCluster;
    numDimensions = dimensions;
    standardDeviation = stdDev;
    numInitializations = initializations;
}

void KMeans::run() {
    generateData();
    computeGroundTruthError();
    kMeansAlgorithm();
}

void KMeans::generateData() {
    // Randomly generate cluster centers
    QRandomGenerator randGen(QDateTime::currentMSecsSinceEpoch());

    startPoints = Eigen::MatrixXd::Random(numClusters, numDimensions);
    trainData.resize(numClusters * numPointsPerCluster, numDimensions);
    testData.resize(numClusters * numPointsPerCluster, numDimensions);
    groundTruthLabels.resize(numClusters * numPointsPerCluster);

    int index = 0;
    for (int i = 0; i < numClusters; ++i) {
        Eigen::VectorXd startPoint = startPoints.row(i);
        for (int j = 0; j < numPointsPerCluster; ++j) {
            Eigen::VectorXd noise = Eigen::VectorXd::Zero(numDimensions);
            for (int d = 0; d < numDimensions; ++d) {
                noise(d) = randGen.generateDouble() * standardDeviation;
            }
            trainData.row(index) = startPoint + noise;
            testData.row(index) = startPoint + noise;
            groundTruthLabels(index) = i;
            ++index;
        }
    }
}

void KMeans::computeGroundTruthError() {
    groundTruthError = 0.0;
    for (int i = 0; i < trainData.rows(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        for (int j = 0; j < numClusters; ++j) {
            double dist = (trainData.row(i) - startPoints.row(j)).squaredNorm();
            if (dist < minDist) {
                minDist = dist;
            }
        }
        groundTruthError += minDist;
    }
}

void KMeans::computeFriendMatrix(const Eigen::VectorXi &labels, Eigen::MatrixXi &friendMatrix) {
    int n = labels.size();
    friendMatrix = Eigen::MatrixXi::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (labels(i) == labels(j)) {
                friendMatrix(i, j) = 1;
            }
        }
    }
}

void KMeans::kMeansAlgorithm() {
    bestError = std::numeric_limits<double>::max();
    QRandomGenerator randGen(QDateTime::currentMSecsSinceEpoch());

    for (int init = 0; init < numInitializations; ++init) {
        // Initialize centroids
        Eigen::MatrixXd centroids(numClusters, numDimensions);
        for (int i = 0; i < numClusters; ++i) {
            int idx = randGen.bounded(trainData.rows());
            centroids.row(i) = trainData.row(idx);
        }

        Eigen::VectorXi labels(trainData.rows());
        double prevError = std::numeric_limits<double>::max();

        for (int iter = 0; iter < 100; ++iter) {
            // Assign labels
            for (int i = 0; i < trainData.rows(); ++i) {
                double minDist = std::numeric_limits<double>::max();
                int minIdx = -1;
                for (int j = 0; j < numClusters; ++j) {
                    double dist = (trainData.row(i) - centroids.row(j)).squaredNorm();
                    if (dist < minDist) {
                        minDist = dist;
                        minIdx = j;
                    }
                }
                labels(i) = minIdx;
            }

            // Update centroids
            Eigen::MatrixXd newCentroids = Eigen::MatrixXd::Zero(numClusters, numDimensions);
            Eigen::VectorXi counts = Eigen::VectorXi::Zero(numClusters);
            for (int i = 0; i < trainData.rows(); ++i) {
                newCentroids.row(labels(i)) += trainData.row(i);
                counts(labels(i)) += 1;
            }
            for (int i = 0; i < numClusters; ++i) {
                if (counts(i) > 0) {
                    newCentroids.row(i) /= counts(i);
                } else {
                    // Reinitialize empty centroid
                    int idx = randGen.bounded(trainData.rows());
                    newCentroids.row(i) = trainData.row(idx);
                }
            }

            double error = 0.0;
            for (int i = 0; i < trainData.rows(); ++i) {
                error += (trainData.row(i) - centroids.row(labels(i))).squaredNorm();
            }

            if (std::abs(prevError - error) < 1e-4) {
                break;
            }
            prevError = error;
            centroids = newCentroids;
        }

        if (prevError < bestError) {
            bestError = prevError;
            bestCentroids = centroids;
            bestLabels = labels;
        }
    }

    // Compute similarity measure
    Eigen::MatrixXi FGT, Ftrain, Fintersect;
    computeFriendMatrix(groundTruthLabels, FGT);
    computeFriendMatrix(bestLabels, Ftrain);
    Fintersect = FGT.cwiseProduct(Ftrain);

    double sumFGT = FGT.sum();
    double sumFtrain = Ftrain.sum();
    double sumFintersect = Fintersect.sum();

    similarityMeasure = (sumFintersect / sumFGT + sumFintersect / sumFtrain) / 2.0;
}

double KMeans::getGroundTruthError() const {
    return groundTruthError;
}

double KMeans::getSimilarityMeasure() const {
    return similarityMeasure;
}

double KMeans::getBestError() const {
    return bestError;
}

void KMeans::plotClusters(QWidget *plotWidget) {
    // Use Qt Charts for plotting
    QtCharts::QChartView *chartView = new QtCharts::QChartView(plotWidget);
    QtCharts::QChart *chart = new QtCharts::QChart();

    QVector<QtCharts::QScatterSeries *> seriesList;
    for (int i = 0; i < numClusters; ++i) {
        QtCharts::QScatterSeries *series = new QtCharts::QScatterSeries();
        series->setName(QString("Cluster %1").arg(i + 1));
        seriesList.append(series);
    }

    for (int i = 0; i < trainData.rows(); ++i) {
        int clusterIdx = bestLabels(i);
        double x = trainData(i, 0);
        double y = trainData(i, 1);
        seriesList[clusterIdx]->append(x, y);
    }

    for (auto series : seriesList) {
        chart->addSeries(series);
    }

    chart->createDefaultAxes();
    chart->setTitle("K-Means Clustering Result");
    chartView->setChart(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QHBoxLayout *layout = new QHBoxLayout(plotWidget);
    layout->addWidget(chartView);
    plotWidget->setLayout(layout);
}

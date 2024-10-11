#include "kmeans.h"
#include <QRandomGenerator>
#include <QDateTime>
#include <QtCharts>
#include <cmath>
#include <limits>
#include <algorithm>
#include <QHBoxLayout>

KMeans::KMeans()
    : numClusters(3),
      numPointsPerCluster(50),
      numDimensions(2),
      standardDeviation(0.1),
      numInitializations(10),
      groundTruthError(0.0),
      similarityMeasure(0.0),
      bestError(std::numeric_limits<double>::max()) {
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
    QRandomGenerator *randGen = QRandomGenerator::global();

    // Generate random cluster centers (startPoints)
    startPoints.clear();
    for (int i = 0; i < numClusters; ++i) {
        DataPoint center;
        center.coordinates.resize(numDimensions);
        for (int d = 0; d < numDimensions; ++d) {
            center.coordinates[d] = randGen->generateDouble();
        }
        startPoints.push_back(center);
    }

    // Generate training and testing data
    trainData.clear();
    testData.clear();
    groundTruthLabels.clear();

    for (int i = 0; i < numClusters; ++i) {
        for (int j = 0; j < numPointsPerCluster; ++j) {
            DataPoint point;
            point.coordinates.resize(numDimensions);
            for (int d = 0; d < numDimensions; ++d) {
                double noise = randGen->generateNormal(0.0, standardDeviation);
                point.coordinates[d] = startPoints[i].coordinates[d] + noise;
            }
            trainData.push_back(point);
            testData.push_back(point); // For simplicity, using same data
            groundTruthLabels.push_back(i);
        }
    }
}

void KMeans::computeGroundTruthError() {
    groundTruthError = 0.0;
    for (const auto& point : trainData) {
        double minDist = std::numeric_limits<double>::max();
        for (const auto& center : startPoints) {
            double dist = 0.0;
            for (int d = 0; d < numDimensions; ++d) {
                double diff = point.coordinates[d] - center.coordinates[d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
            }
        }
        groundTruthError += minDist;
    }
}

void KMeans::computeFriendMatrix(const std::vector<int>& labels, std::vector<std::vector<int>>& friendMatrix) {
    int n = labels.size();
    friendMatrix.resize(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (labels[i] == labels[j]) {
                friendMatrix[i][j] = 1;
            }
        }
    }
}

void KMeans::kMeansAlgorithm() {
    bestError = std::numeric_limits<double>::max();
    QRandomGenerator *randGen = QRandomGenerator::global();
    double totalRuntime = 0.0;

    for (int init = 0; init < numInitializations; ++init) {
        // Initialize centroids
        std::vector<DataPoint> centroids(numClusters);
        for (int i = 0; i < numClusters; ++i) {
            int idx = randGen->bounded(static_cast<int>(trainData.size()));
            centroids[i] = trainData[idx];
        }

        std::vector<int> labels(trainData.size(), -1);
        double prevError = std::numeric_limits<double>::max();

        for (int iter = 0; iter < 100; ++iter) {
            // Assign labels
            for (size_t i = 0; i < trainData.size(); ++i) {
                double minDist = std::numeric_limits<double>::max();
                int minIdx = -1;
                for (int j = 0; j < numClusters; ++j) {
                    double dist = 0.0;
                    for (int d = 0; d < numDimensions; ++d) {
                        double diff = trainData[i].coordinates[d] - centroids[j].coordinates[d];
                        dist += diff * diff;
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        minIdx = j;
                    }
                }
                labels[i] = minIdx;
            }

            // Update centroids
            std::vector<DataPoint> newCentroids(numClusters);
            std::vector<int> counts(numClusters, 0);

            for (int i = 0; i < numClusters; ++i) {
                newCentroids[i].coordinates.resize(numDimensions, 0.0);
            }

            for (size_t i = 0; i < trainData.size(); ++i) {
                int label = labels[i];
                for (int d = 0; d < numDimensions; ++d) {
                    newCentroids[label].coordinates[d] += trainData[i].coordinates[d];
                }
                counts[label] += 1;
            }

            for (int i = 0; i < numClusters; ++i) {
                if (counts[i] > 0) {
                    for (int d = 0; d < numDimensions; ++d) {
                        newCentroids[i].coordinates[d] /= counts[i];
                    }
                } else {
                    // Reinitialize empty centroid
                    int idx = randGen->bounded(static_cast<int>(trainData.size()));
                    newCentroids[i] = trainData[idx];
                }
            }

            double error = 0.0;
            for (size_t i = 0; i < trainData.size(); ++i) {
                int label = labels[i];
                double dist = 0.0;
                for (int d = 0; d < numDimensions; ++d) {
                    double diff = trainData[i].coordinates[d] - centroids[label].coordinates[d];
                    dist += diff * diff;
                }
                error += dist;
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
    std::vector<std::vector<int>> FGT, Ftrain, Fintersect;
    computeFriendMatrix(groundTruthLabels, FGT);
    computeFriendMatrix(bestLabels, Ftrain);

    int n = groundTruthLabels.size();
    Fintersect.resize(n, std::vector<int>(n, 0));
    int sumFGT = 0, sumFtrain = 0, sumFintersect = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Fintersect[i][j] = FGT[i][j] * Ftrain[i][j];
            sumFGT += FGT[i][j];
            sumFtrain += Ftrain[i][j];
            sumFintersect += Fintersect[i][j];
        }
    }

    similarityMeasure = (static_cast<double>(sumFintersect) / sumFGT + static_cast<double>(sumFintersect) / sumFtrain) / 2.0;
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

    std::vector<QtCharts::QScatterSeries *> seriesList(numClusters);
    for (int i = 0; i < numClusters; ++i) {
        QtCharts::QScatterSeries *series = new QtCharts::QScatterSeries();
        series->setName(QString("Cluster %1").arg(i + 1));
        seriesList[i] = series;
    }

    for (size_t i = 0; i < trainData.size(); ++i) {
        int clusterIdx = bestLabels[i];
        double x = trainData[i].coordinates[0];
        double y = trainData[i].coordinates[1];
        seriesList[clusterIdx]->append(x, y);
    }

    for (auto series : seriesList) {
        chart->addSeries(series);
    }

    chart->createDefaultAxes();
    chart->setTitle("K-Means Clustering Result");
    chart->legend()->setAlignment(Qt::AlignBottom);
    chartView->setChart(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    // Clear previous layout if any
    if (plotWidget->layout() != nullptr) {
        delete plotWidget->layout();
    }

    QHBoxLayout *layout = new QHBoxLayout(plotWidget);
    layout->addWidget(chartView);
    plotWidget->setLayout(layout);
}

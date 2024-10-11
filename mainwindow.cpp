#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "kmeans.h"

#include <QMessageBox>
#include <QElapsedTimer>
#include <random>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow) {
    ui->setupUi(this);
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::on_runButton_clicked() {
    // Get parameters from UI
    bool ok;
    int numClusters = ui->clustersSpinBox->value();
    int numPoints = ui->pointsSpinBox->value();
    int numDimensions = ui->dimensionsSpinBox->value();
    double stdDev = ui->stdDevSpinBox->value();
    int numInitializations = ui->initializationsSpinBox->value();

    // Validate inputs
    if (numClusters <= 0 || numPoints <= 0 || numDimensions <= 0 || stdDev <= 0) {
        QMessageBox::warning(this, "Input Error", "All input values must be positive.");
        return;
    }

    // Generate data and run k-means
    KMeans kmeans;
    kmeans.setParameters(numClusters, numPoints, numDimensions, stdDev, numInitializations);

    QElapsedTimer timer;
    timer.start();
    kmeans.run();
    qint64 elapsedTime = timer.elapsed();

    // Display statistics
    QString output;
    output += QString("Ground truth error: %1\n").arg(kmeans.getGroundTruthError(), 0, 'f', 4);
    output += QString("Cluster similarity measure S: %1\n").arg(kmeans.getSimilarityMeasure(), 0, 'f', 4);
    output += QString("Best training error at convergence: %1\n").arg(kmeans.getBestError(), 0, 'f', 4);
    output += QString("Total runtime: %1 seconds\n").arg(elapsedTime / 1000.0, 0, 'f', 4);
    output += QString("Average runtime per initialization: %1 seconds\n").arg((elapsedTime / 1000.0) / numInitializations, 0, 'f', 4);

    ui->textBrowser->setText(output);

    // Plot results if dimensions are 2 or 3
    if (numDimensions == 2 || numDimensions == 3) {
        kmeans.plotClusters(ui->plotWidget);
    } else {
        QMessageBox::information(this, "Plotting", "Plotting is only available for 2D or 3D data.");
    }
}

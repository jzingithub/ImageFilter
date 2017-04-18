#include "mainwindow.h"

#include <QApplication>
#include <QWidget>
#include <QImage>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QFileDialog>

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include "base/convert.h"

using namespace cv;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

    QWidget *wn = new QWidget;
    wn->setWindowTitle("disp image");

    QString filename = QFileDialog::getOpenFileName(0, "Open File", "", "*.jpg *.png *.bmp", 0);
    if (filename.isNull()) {
        return -1;
    }

    Mat image = imread(filename.toStdString().c_str(), 1);
    QImage img = jz::convert::MatToQImage(image);

    QLabel *label = new QLabel("", 0);
    label->setPixmap(QPixmap::fromImage(img));

    QPushButton *bnt = new QPushButton("Quit");
    QObject::connect(bnt, SIGNAL(clicked()), &a, SLOT(quit()));

    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(label);
    layout->addWidget(bnt);
    wn->setLayout(layout);

    wn->show();

    return a.exec();
}

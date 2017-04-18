#ifndef IMAGE_FILTER_CONVERT_H
#define IMAGE_FILTER_CONVERT_H

#include <QImage>
#include <opencv/cv.h>

namespace jz
{
    namespace convert
    {
        enum MatColorOrder
        {
            MCO_BGR,
            MCO_RGB,
            MCO_BGRA = MCO_BGR,
            MCO_RGBA = MCO_RGB,
            MCO_ARGB
        };

        // convert cv::Mat to QImage
        QImage MatToQImage(const cv::Mat& mat_image,
                           MatColorOrder mat_color_order = MatColorOrder::MCO_BGR,
                           QImage::Format format_hint = QImage::Format_Invalid);

        // convert cv::Mat to QImage without data copy
        QImage MatToQImage_Shared(const cv::Mat& mat, QImage::Format format_hint);

        // convert QImage to cv::Mat
        cv::Mat QImageToMat(const QImage& qimage,
                            int required_mat_type,
                            MatColorOrder required_order);

        // convert QImage to cv::Mat without data copy
        cv::Mat QImageToMat_Shared(const QImage& qimage, MatColorOrder* ptr_order);
    }
}

#endif

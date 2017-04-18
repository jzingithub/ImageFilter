#include "base/convert.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace jz
{

namespace convert
{
    // anonymous namespace include help functions for internal use
    namespace
    {
        cv::Mat ARGBToBGRA(const cv::Mat& mat)
        {
            Q_ASSERT(4 == mat.channels());

            cv::Mat converted_mat(mat.rows,
                                  mat.cols,
                                  mat.type());
            int from_to[] = { 0, 3, 1, 2, 2, 1, 3, 0 };
            cv::mixChannels(&mat,
                            1,
                            &converted_mat,
                            1,
                            from_to,
                            4);
            return converted_mat;
        }

        QImage::Format FindClosestFormat(QImage::Format format_hint)
        {
            QImage::Format format;
            switch (format_hint)
            {
            case QImage::Format_Indexed8:
            case QImage::Format_RGB32:
            case QImage::Format_ARGB32:
            case QImage::Format_ARGB32_Premultiplied:
            case QImage::Format_RGB888:
            case QImage::Format_RGBX8888:
            case QImage::Format_RGBA8888:
            case QImage::Format_RGBA8888_Premultiplied:
            case QImage::Format_Alpha8:
            case QImage::Format_Grayscale8:
                format = format_hint;
                break;

            case QImage::Format_Mono:
            case QImage::Format_MonoLSB:
                format = QImage::Format_Indexed8;
                break;

            case QImage::Format_RGB16:
                format = QImage::Format_RGB32;
                break;

            case QImage::Format_RGB444:
            case QImage::Format_RGB555:
            case QImage::Format_RGB666:
                format = QImage::Format_RGB888;
                break;

            case QImage::Format_ARGB4444_Premultiplied:
            case QImage::Format_ARGB6666_Premultiplied:
            case QImage::Format_ARGB8555_Premultiplied:
            case QImage::Format_ARGB8565_Premultiplied:
                format = QImage::Format_ARGB32_Premultiplied;
                break;

            default:
                format = QImage::Format_ARGB32;
                break;
            }

            return format;
        }

        inline MatColorOrder GetColorOrderOfRGB32Format()
        {
        #if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
                return MCO_BGRA;
        #else
                return MCO_ARGB;
        #endif
        }

        cv::Mat AdjustChannelsOrder(const cv::Mat& src_mat,
                                    MatColorOrder src_order,
                                    MatColorOrder target_order)
        {
            Q_ASSERT(4 == src_mat.channels());
            if (src_order == target_order) { return src_mat.clone(); }

            cv::Mat output_mat;
            if ((MCO_ARGB == src_order && MCO_BGRA == target_order) ||
                (MCO_BGRA == src_order && MCO_ARGB == target_order))
            {
                // ARGB <-> BGRA
                output_mat = ARGBToBGRA(src_mat);
            }
            else if (MCO_ARGB == src_order && MCO_RGBA == target_order)
            {
                // ARGB -> RGBA
                output_mat = cv::Mat(src_mat.rows,
                                     src_mat.cols,
                                     src_mat.type());
                int from_to[] = { 0, 3, 1, 0, 2, 1, 3, 2 };
                cv::mixChannels(&src_mat,
                                1,
                                &output_mat,
                                1,
                                from_to,
                                4);
            }
            else if (MCO_RGBA == src_order && MCO_ARGB == target_order)
            {
                // RGBA -> ARGB
                output_mat = cv::Mat(src_mat.rows,
                                     src_mat.cols,
                                     src_mat.type());
                int from_to[] = { 0, 1, 1, 2, 2, 3, 3, 0 };
                cv::mixChannels(&src_mat,
                                1,
                                &output_mat,
                                1,
                                from_to,
                                4);
            }
            else
            {
                cv::cvtColor(src_mat,
                             output_mat,
                             CV_BGRA2RGBA);
            }

            return output_mat;
        }

    } // end of anonymous namespace

    QImage MatToQImage(const cv::Mat& mat_image,
                       MatColorOrder mat_color_order,
                       QImage::Format format_hint)
    {
        Q_ASSERT(1 == mat_image.channels() ||
                 3 == mat_image.channels() ||
                 4 == mat_image.channels());
        Q_ASSERT(CV_8U == mat_image.depth() ||
                 CV_16U == mat_image.depth() ||
                 CV_32F == mat_image.depth());

        if (mat_image.empty()) { return QImage(); }
        // adjust mat_image's channels if needed and find proper QImage format
        auto format = QImage::Format_Invalid;
        cv::Mat mat_adjusted_channels;
        if (1 == mat_image.channels())
        {
            format = format_hint;
            if (format_hint != QImage::Format_Indexed8 &&
                format_hint != QImage::Format_Alpha8 &&
                format_hint != QImage::Format_Grayscale8)
            {
                format = QImage::Format_Indexed8;
            }
        }
        else if (3 == mat_image.channels())
        {
            format = QImage::Format_RGB888;
            if (MCO_BGR == mat_color_order)
            {
                cv::cvtColor(mat_image,
                             mat_adjusted_channels,
                             CV_BGR2RGB);
            }
        }
        else if (4 == mat_image.channels())
        {
            format = FindClosestFormat(format_hint);
            if (QImage::Format_RGB32 != format &&
                QImage::Format_ARGB32 != format &&
                QImage::Format_ARGB32_Premultiplied != format &&
                QImage::Format_RGBX8888 != format &&
                QImage::Format_RGBA8888 != format &&
                QImage::Format_RGBA8888_Premultiplied != format)
            {
                format = (mat_color_order == MCO_RGBA) ?
                       QImage::Format_RGBA8888 : QImage::Format_ARGB32;
            }

            // channel order required by the target QImage
            MatColorOrder required_order = GetColorOrderOfRGB32Format();
            if (QImage::Format_RGBX8888 == format_hint ||
                QImage::Format_RGBA8888 == format_hint ||
                QImage::Format_RGBA8888_Premultiplied == format_hint)
            {
                required_order = MCO_RGBA;
            }
            if (mat_color_order != required_order)
            {
                mat_adjusted_channels = AdjustChannelsOrder(mat_image,
                                                            mat_color_order,
                                                            required_order);
            }
        }

        if (mat_adjusted_channels.empty())
        {
            mat_adjusted_channels = mat_image;
        }

        // adjust mat depth if needed
        auto mat_adjusted_depth = mat_adjusted_channels;
        if (CV_8U != mat_image.depth())
        {
            mat_adjusted_channels.convertTo(mat_adjusted_depth,
                                            CV_8UC(mat_adjusted_channels.channels()),
                                            CV_16U == mat_image.depth() ? 1/255.0 : 255.0);
        }

        // see if it is needed to convert the image to the format specified by format_hint
        QImage qimage = MatToQImage_Shared(mat_adjusted_depth, format);
        if (format == format_hint || format_hint == QImage::Format_Invalid)
        {
            return qimage.copy();
        }
        else
        {
            return qimage.convertToFormat(format_hint);
        }
    }

    // convert cv::Mat to QImage without data copy
    QImage MatToQImage_Shared(const cv::Mat& mat, QImage::Format format_hint)
    {
        Q_ASSERT(CV_8UC1 == mat.type() ||
                 CV_8UC3 == mat.type() ||
                 CV_8UC4 == mat.type());

        if (mat.empty()) { return QImage(); }

        // adjust format_hint if needed
        if (CV_8UC1 == mat.type())
        {
            if (QImage::Format_Indexed8 != format_hint &&
                QImage::Format_Alpha8 != format_hint &&
                QImage::Format_Grayscale8 != format_hint)
            {
                format_hint = QImage::Format_Indexed8;
            }
        }
        else if (CV_8UC3 == mat.type())
        {
            format_hint = QImage::Format_RGB888;
        }
        else if (CV_8UC4 == mat.type())
        {
            if (QImage::Format_RGB32 != format_hint &&
                QImage::Format_ARGB32 != format_hint &&
                QImage::Format_ARGB32_Premultiplied != format_hint &&
                QImage::Format_RGBX8888 != format_hint &&
                QImage::Format_RGBA8888 != format_hint &&
                QImage::Format_RGBA8888_Premultiplied != format_hint)
            {
                format_hint = QImage::Format_ARGB32;
            }
        }

        // mat.step instead of mat.cols * mat.channels()
        // mat.step tell QImage how many bytes per row
        // without it, compiler won't complain
        // but some image won't show properly
        QImage qimage(mat.data,
                      mat.cols,
                      mat.rows,
                      mat.step,
                      format_hint);

        // see if it is needed to support user-customed colortable
        if (QImage::Format_Indexed8 == format_hint)
        {
            QVector<QRgb> color_table;
            for (int i = 0; i < 256; ++i)
            {
                color_table.append(qRgb(i, i, i));
            }
            qimage.setColorTable(color_table);
        }

        return qimage;

    }

    cv::Mat QImageToMat(const QImage& qimage,
                        int required_mat_type,
                        MatColorOrder required_order)
    {
        int target_depth = CV_MAT_DEPTH(required_mat_type);
        int target_channels = CV_MAT_CN(required_mat_type);
        Q_ASSERT(CV_CN_MAX == target_channels ||
                 1 == target_channels ||
                 3 == target_channels ||
                 4 == target_channels);
        Q_ASSERT(CV_8U == target_depth ||
                 CV_16U == target_depth ||
                 CV_32F == target_depth);

        if (qimage.isNull()) { return cv::Mat(); }

        // find the closest image format that can be used in QImageToMat_Shared()
        auto format = FindClosestFormat(qimage.format());
        QImage converted_qimage = (qimage.format() == format) ?
                    qimage : qimage.convertToFormat(format);

        MatColorOrder src_order;
        cv::Mat shared_mat = QImageToMat_Shared(converted_qimage, &src_order);

        // adjust mat channels if needed
        cv::Mat mat_adjusted_channels;
        const float max_alpha = (CV_8U == target_depth)
                ? 255 : (CV_16U == target_depth ? 65535 : 1.0);
        if (CV_CN_MAX == target_channels)
        {
            target_channels = shared_mat.channels();
        }

        switch (target_channels)
        {
        case 1:
            if (3 == shared_mat.channels())
            {
                cv::cvtColor(shared_mat,
                             mat_adjusted_channels,
                             CV_RGB2GRAY);
            }
            else if (4 == shared_mat.channels())
            {
                if (MCO_BGRA == src_order)
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_BGRA2GRAY);
                }
                else if (MCO_RGBA == src_order)
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_RGBA2GRAY);
                }
                else // MCO_ARGB
                {
                    cv::cvtColor(ARGBToBGRA(shared_mat),
                                 mat_adjusted_channels,
                                 CV_BGRA2GRAY);
                }
            }
            break; // end of case 1
        case 3:
            if (1 == shared_mat.channels())
            {
                cv::cvtColor(shared_mat,
                             mat_adjusted_channels,
                             required_order == MCO_BGR ? CV_GRAY2BGR : CV_GRAY2RGB);
            }
            else if (3 == shared_mat.channels())
            {
                if (required_order != src_order)
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_RGB2BGR);
                }
            }
            else if (4 == shared_mat.channels())
            {
                if (MCO_ARGB == src_order)
                {
                    mat_adjusted_channels = cv::Mat(shared_mat.rows,
                                                    shared_mat.cols,
                                                    CV_MAKE_TYPE(shared_mat.type(), 3));
                    int ARGB2RGB[] = { 1, 0, 2, 1, 3, 2 };
                    int ARGB2BGR[] = { 1, 2, 2, 1, 3, 0 };
                    cv::mixChannels(&shared_mat,
                                    1,
                                    &mat_adjusted_channels,
                                    1,
                                    MCO_BGR == required_order ? ARGB2BGR : ARGB2RGB,
                                    3);
                }
                else if (MCO_BGRA == src_order)
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 MCO_BGR == required_order ? CV_BGRA2BGR : CV_BGRA2RGB);
                }
                else // RGBA
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 MCO_BGR == required_order ? CV_RGBA2BGR : CV_RGBA2RGB);
                }
            }
            break; // end of case 3
        case 4:
            if (1 == shared_mat.channels())
            {
                if (MCO_ARGB == required_order)
                {
                    cv::Mat alpha_mat(shared_mat.rows,
                                      shared_mat.cols,
                                      CV_MAKE_TYPE(shared_mat.type(), 1),
                                      cv::Scalar(max_alpha));
                    mat_adjusted_channels = cv::Mat(shared_mat.rows,
                                                    shared_mat.cols,
                                                    CV_MAKE_TYPE(shared_mat.type(), 4));
                    cv::Mat mat_array[] = { alpha_mat, shared_mat };
                    int from_to[] = { 0, 0, 1, 1, 1, 2, 1, 3 };
                    cv::mixChannels(mat_array,
                                    2,
                                    &mat_adjusted_channels,
                                    1,
                                    from_to,
                                    4);
                }
                else if (MCO_RGBA == required_order)
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_GRAY2RGBA);
                }
                else // MCO_BGRA
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_RGB2BGRA);
                }
            }
            else if (3 == shared_mat.channels())
            {
                if (MCO_ARGB == required_order)
                {
                    cv::Mat alpha_mat(shared_mat.rows,
                                      shared_mat.cols,
                                      CV_MAKE_TYPE(shared_mat.type(), 1),
                                      cv::Scalar(max_alpha));
                    mat_adjusted_channels = cv::Mat(shared_mat.rows,
                                                    shared_mat.cols,
                                                    CV_MAKE_TYPE(shared_mat.type(), 4));
                    cv::Mat mat_array[] = { alpha_mat, shared_mat };
                    int from_to[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
                    cv::mixChannels(mat_array,
                                    2,
                                    &mat_adjusted_channels,
                                    1,
                                    from_to,
                                    4);
                }
                else if (MCO_RGBA == required_order)
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_RGB2RGBA);
                }
                else // MCO_BGRA
                {
                    cv::cvtColor(shared_mat,
                                 mat_adjusted_channels,
                                 CV_RGB2BGRA);
                }
            }
            else if (4 == shared_mat.channels())
            {
                if (src_order != required_order)
                {
                    mat_adjusted_channels = AdjustChannelsOrder(shared_mat,
                                                                src_order,
                                                                required_order);
                }
            }
            break; // end of case 4
        default:
            break;
        } // end of switch (target_channels)

        // adjust depth if needed
        if (CV_8U == target_depth)
        {
            return mat_adjusted_channels.empty() ?
                        shared_mat.clone() : mat_adjusted_channels;
        }

        if (mat_adjusted_channels.empty()) { mat_adjusted_channels = shared_mat; }
        cv::Mat mat_adjusted_depth;
        mat_adjusted_channels.convertTo(mat_adjusted_depth,
                                        CV_MAKE_TYPE(target_depth, mat_adjusted_channels.channels()),
                                        CV_16U == target_depth ? 255.0 : 1 / 255.0);

        return mat_adjusted_depth;
    }

    // convert QImage to cv::Mat without data copy
    cv::Mat QImageToMat_Shared(const QImage& qimage, MatColorOrder* ptr_order)
    {
        if (qimage.isNull()) { return cv::Mat(); }

        switch (qimage.format())
        {
        case QImage::Format_Indexed8:
            break;
        case QImage::Format_RGB888:
            if (ptr_order) { *ptr_order = MCO_RGB; }
            break;
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied:
            if (ptr_order) { *ptr_order = GetColorOrderOfRGB32Format(); }
            break;
        case QImage::Format_RGBX8888:
        case QImage::Format_RGBA8888:
        case QImage::Format_RGBA8888_Premultiplied:
            if (ptr_order) { *ptr_order = MCO_RGBA; };
            break;
        case QImage::Format_Alpha8:
        case QImage::Format_Grayscale8:
            break;
        default:
            return cv::Mat();
        }

        return cv::Mat(qimage.height(),
                       qimage.width(),
                       CV_8UC(qimage.depth() / 8),
                       const_cast<uchar*>(qimage.bits()),
                       qimage.bytesPerLine());
    }

} // end of namespace 'jz::convert'

} // end of namespace jz

#-------------------------------------------------
#
# Project created by QtCreator 2017-04-18T13:54:33
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageFilter
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += main.cpp\
        mainwindow.cpp \
    base/convert.cpp

HEADERS  += mainwindow.h \
    base/convert.h

FORMS    += mainwindow.ui

# added by jz
INCLUDEPATH += ..\ThirdParty\opencv\include

# don not forget to add the path of corresoponding DLLs to the system environment variable
# copy libs to the directory generated by compiler if using relative path
LIBS += .\lib\opencv\libopencv_calib3d310.dll.a\
        .\lib\opencv\libopencv_core310.dll.a\
        .\lib\opencv\libopencv_features2d310.dll.a\
        .\lib\opencv\libopencv_flann310.dll.a\
        .\lib\opencv\libopencv_highgui310.dll.a\
        .\lib\opencv\libopencv_imgcodecs310.dll.a\
        .\lib\opencv\libopencv_imgproc310.dll.a\
        .\lib\opencv\libopencv_ml310.dll.a\
        .\lib\opencv\libopencv_objdetect310.dll.a\
        .\lib\opencv\libopencv_photo310.dll.a\
        .\lib\opencv\libopencv_shape310.dll.a\
        .\lib\opencv\libopencv_stitching310.dll.a\
        .\lib\opencv\libopencv_superres310.dll.a\
        .\lib\opencv\libopencv_ts310.a\
        .\lib\opencv\libopencv_video310.dll.a\
        .\lib\opencv\libopencv_videoio310.dll.a\
        .\lib\opencv\libopencv_videostab310.dll.a

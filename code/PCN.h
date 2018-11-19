#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

#define M_PI  3.14159265358979323846
#define CLAMP(x, l, u)  ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
#define EPS  1e-5

struct Window
{
    int x, y, width;
    float angle, score;
    Window(int x_, int y_, int w_, float a_, float s_)
        : x(x_), y(y_), width(w_), angle(a_), score(s_)
    {}
};

cv::Point RotatePoint(int x, int y, float centerX, float centerY, float angle);
void DrawLine(cv::Mat img, std::vector<cv::Point> pointList);

void DrawFace(cv::Mat img, Window face);


cv::Mat CropFace(cv::Mat img, Window face, int cropSize);


class PCN
{
public:
    PCN(std::string model, std::string net1, std::string net2, std::string net3);
    void SetMinFaceSize(int minFace);
    void SetScoreThresh(float thresh1, float thresh2, float thresh3);
    void SetImagePyramidScaleFactor(float factor);
    void SetVideoSmooth(bool smooth);
    std::vector<Window> DetectFace(cv::Mat img);

private:
    void* impl_;
};

#endif

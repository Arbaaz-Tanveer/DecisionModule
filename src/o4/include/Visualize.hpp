#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "Coordinate.hpp"

class Visualize {
   private:
    // Field Dimensions
    const double A = 22, B = 14, C = 6.9, D = 3.9, E = 2.25, F = .75, G = .75,
                 H = 4, I = 3.6, J = .15, K = .125, L = 1, M = 1, N = 8, O = 1,
                 P = .5, Q = 3.5, r = 0.5,
                 s = 2.4;  // r = width of goalpost, s = length of goalpost
    double resX, resY;     // Image Resolution
    double scale;          // Scaling Factor

    //  void drawArrow(cv::Mat &image, const cv::Point &pStart,
    //                 const cv::Point &pEnd, const cv::Scalar &color,
    //                 int thickness, int line_type, int shift, double
    //                 tipLength);

    void drawHeadingV(cv::Mat &image, const cv::Point &center,
                      double headingAngle, const cv::Scalar &color,
                      int thickness, int line_type, int shift,
                      double tipLength);

   public:
    Visualize(double resolutionX);
    void visualizeGame(std::vector<Point2D> &path, Point2D &nowPos, int count,
                       double yaw, std::vector<Point2D> &obstacles,
                       Point2D &ball);
};

#endif
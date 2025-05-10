#include "Visualize.hpp"

Visualize::Visualize(double resolutionX) {
    double fieldX = A + 2 * L, fieldY = (B + 2 * L);
    resX = resolutionX;
    resY = resX * (fieldY / fieldX);
    scale = resX / fieldX;
}

// Function to draw a V shape representing the heading on an OpenCV image
void Visualize::drawHeadingV(cv::Mat &image, const cv::Point &center,
                             double headingAngle, const cv::Scalar &color,
                             int thickness, int line_type, int shift,
                             double tipLength) {
    const double tipSize = tipLength;

    const double angle1 = headingAngle + CV_PI / 9;
    const double angle2 = headingAngle - CV_PI / 9;

    cv::Point p1, p2;

    p1.x = static_cast<int>(center.x + tipSize * std::cos(angle1));
    p1.y = static_cast<int>(center.y + tipSize * std::sin(angle1));
    cv::line(image, center, p1, color, thickness, line_type, shift);

    p2.x = static_cast<int>(center.x + tipSize * std::cos(angle2));
    p2.y = static_cast<int>(center.y + tipSize * std::sin(angle2));
    cv::line(image, center, p2, color, thickness, line_type, shift);
}

void Visualize::visualizeGame(std::vector<Point2D> &path, Point2D &nowPos,
                              int count, double yaw,
                              std::vector<Point2D> &obstacles, Point2D &ball) {
    cv::Mat image(resY, resX, CV_8UC3,
                  cv::Scalar(20, 170, 30));  // Official Color: (0, 150, 10)
    double centerX = image.cols / 2.0;
    double centerY = image.rows / 2.0;

    // drawing arena
    double a = A * scale;
    double b = B * scale;
    // Calculate the position of the top-left corner for center alignment
    int x = (image.cols - a) / 2;  // x-coordinate for center alignment
    int y = (image.rows - b) / 2;  // y-coordinate for center alignment

    // Draw a vertical line passing through the center of the rectangle
    cv::Point center1(image.cols / 2, image.rows / 2 - b / 2);
    cv::Point center2(image.cols / 2, image.rows / 2 + b / 2);
    cv::Scalar color(255, 255, 255);  // white color for the line
    cv::line(image, center1, center2, color, K * scale);

    // Draw a white circle with its circumference in the center of the image
    int radius = H * scale / 2;  // Adjust the radius as needed
    cv::circle(image, cv::Point(centerX, centerY), radius, color, K * scale);

    // Draw the rectangle on the image
    cv::rectangle(image, cv::Rect(x, y, a, b), color, K * scale);

    // penalty area
    //  Define the dimensions of the new rectangle attached to the right
    //  vertical edge
    int penaltyAreaWidth = E * scale;
    int penaltyAreaHeight = 2 * Q * scale;

    int goalAreaWidth = F * scale;
    int goalAreaHeight = D * scale;
    // Calculate the position of the top-left corner for the goal rectangle
    int goalX_r =
        (x + a) - (goalAreaWidth);  // Right edge of the main rectangle
    int goalY_r = y + (b - goalAreaHeight) / 2;  // Center vertically
    int goalX_l = x;  // Left edge of the main rectangle
    int goalY_l = y + (b - goalAreaHeight) / 2;

    // goal area
    int penaltyX_r =
        (x + a) - (penaltyAreaWidth);  // Right edge of the main rectangle
    int penaltyY_r = y + (b - penaltyAreaHeight) / 2;  // Center vertically
    int penaltyX_l = x;  // Left edge of the main rectangle
    int penaltyY_l = y + (b - penaltyAreaHeight) / 2;

    // Draw the new rectangle on the image
    cv::rectangle(
        image,
        cv::Rect(penaltyX_r, penaltyY_r, penaltyAreaWidth, penaltyAreaHeight),
        color, K * scale);
    cv::rectangle(
        image,
        cv::Rect(penaltyX_l, penaltyY_l, penaltyAreaWidth, penaltyAreaHeight),
        color, K * scale);

    cv::rectangle(image,
                  cv::Rect(goalX_r, goalY_r, goalAreaWidth, goalAreaHeight),
                  color, K * scale);
    cv::rectangle(image,
                  cv::Rect(goalX_l, goalY_l, goalAreaWidth, goalAreaHeight),
                  color, K * scale);

    // adding corners
    //  Define the dimensions of the quarter circles
    int radiusCorner = G * scale;  // Adjust the radius as needed

    // Calculate the center points for the four corners
    cv::Point topLeft(x, y);
    cv::Point topRight(x + a, y);
    cv::Point bottomLeft(x, y + b);
    cv::Point bottomRight(x + a, y + b);

    // Define the color of the quarter circles (BGR format)
    cv::Scalar circleColor(255, 255, 255);  // White color

    // Draw quarter circles at each corner
    cv::ellipse(image, topLeft, cv::Size(radiusCorner, radiusCorner), 0, 0, 90,
                circleColor, K * scale);
    cv::ellipse(image, topRight, cv::Size(radiusCorner, radiusCorner), 0, -180,
                -270, circleColor, K * scale);
    cv::ellipse(image, bottomLeft, cv::Size(radiusCorner, radiusCorner), 0, -90,
                0, circleColor, K * scale);
    cv::ellipse(image, bottomRight, cv::Size(radiusCorner, radiusCorner), 0,
                180, 270, circleColor, K * scale);

    // adding goal post
    //  Define the dimensions of the additional rectangle

    double R = r * scale;
    double S = s * scale;
    // Calculate the position of the top-left corner for the additional
    // rectangle
    int newXLeft = x - R;            // To the left of the main rectangle
    int newYLeft = y - (S - b) / 2;  // Align vertically with the main rectangle

    int newXRight = x + a;  // To the right of the main rectangle
    int newYRight =
        y - (S - b) / 2;  // Align vertically with the main rectangle

    // Draw the additional rectangles on both sides of the main rectangle
    cv::rectangle(image, cv::Rect(newXLeft, newYLeft, R, S), color, K * scale);
    cv::rectangle(image, cv::Rect(newXRight, newYRight, R, S), color,
                  K * scale);

    // Draw horizontal lines dividing the additional rectangles
    int lineYLeft =
        newYLeft + S / 2;  // Y-coordinate for the left horizontal line
    int lineYRight =
        newYRight + S / 2;  // Y-coordinate for the right horizontal line

    cv::Point lineStartLeft(newXLeft, lineYLeft);
    cv::Point lineEndLeft(newXLeft + R, lineYLeft);

    cv::Point lineStartRight(newXRight, lineYRight);
    cv::Point lineEndRight(newXRight + R, lineYRight);

    cv::line(image, lineStartLeft, lineEndLeft, color, K * scale);
    cv::line(image, lineStartRight, lineEndRight, color, K * scale);

    cv::Point robotPose((nowPos.x * scale) + centerX,
                        -(nowPos.y * scale) + centerY);

    // converting obstacles and ball Point2D type to Point type
    std::vector<cv::Point> obs;
    for (auto &it : obstacles) {
        cv::Point point(it.x * scale + centerX, -it.y * scale + centerY);
        obs.push_back(point);
    }

    for (auto &it : path) {
        cv::circle(
            image,
            cv::Point2d(it.x * scale + centerX, -(it.y * scale) + centerY), 5,
            cv::Scalar(0, 0, 0), -1);
    }

    // Draw the robot base (you can replace this with your robot's actual base)
    cv::circle(image, robotPose, 40 * scale / 100, cv::Scalar(0, 150, 255), -1);
    // cv::circle(image, obs[0], 25, cv::Scalar(255, 0, 0), -1);
    drawHeadingV(image, robotPose, -yaw, cv::Scalar(60, 60, 180), 5, 8, 0,
                 40 * scale / 100);

    // Draw obstacles
    for (int i = 0; i < obstacles.size(); i++) {
        cv::circle(image, obs[i], 40 * scale / 100, cv::Scalar(255, 225, 0),
                   -1);
        drawHeadingV(image, obs[i], -obstacles[i].theta,
                     cv::Scalar(128, 0, 128), 5, 8, 0, 40 * scale / 100);
    }

    cv::Point BALL(ball.x * scale + centerX, -(ball.y * scale) + centerY);
    // draw ball
    cv::circle(image, BALL, 11.25 * scale / 100, cv::Scalar(255, 255, 255), -1);

    // Add black outline to the football
    cv::circle(image, BALL, 11.25 * scale / 100, cv::Scalar(0, 0, 0),
               4 * scale / 100);  // Draw a black outline

    // Draw black line from robot to the destination
    cv::line(image, robotPose,
             cv::Point(path[count].x * scale + centerX,
                       -(path[count].y * scale) + centerY),
             cv::Scalar(0, 0, 0), 2);

    cv::imshow("Robot 4 omni visual", image);
    cv::waitKey(1);
}

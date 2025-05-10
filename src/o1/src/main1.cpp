#include <iostream>
#include <cmath>
#include <vector>

#include "Coordinate.hpp"
#include "PathPlanner.hpp"
#include "Visualize.hpp"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std;

double min(double a, double b) { return a < b ? a : b; }
double max(double a, double b) { return a > b ? a : b; }

double set_angle(double theta)
{

    while (theta > M_PI)
    {
        theta -= 2 * M_PI;
    }
    while (theta < -M_PI)
    {
        theta += 2 * M_PI;
    }
    return theta;
}

Visualize *window = new Visualize(800);
double runTime = 0.01;
double A = 22, B = 14;
optimalPlanner plannerType = PLANNER_RRTSTAR;
planningObjective objectiveType = OBJECTIVE_PATHLENGTH;

long long int iteration_count = 0;
int flag = 0;
double obs_size = 0.7;
double max_angle = 0.5;
double max_acceleration = 1;
double max_speed = 2;

std::vector<Point2D> obs(9), targetPos;
Point2D nowPos(0, 0, 0), finalPos(0, 0, 0), ballPos(0, 0, 0);
std::vector<pair<double, double>> points;

int findclosestpoint(std::vector<Point2D> &targetPos, Point2D &nowPos)
{
    double min_dist = 1000000;
    int idx = 0;
    for (int i = 0; i < targetPos.size(); i++) // excluding last point and start
    {
        if (sqrtl((targetPos[i].x - nowPos.x) * (targetPos[i].x - nowPos.x) +
                  (targetPos[i].y - nowPos.y) * (targetPos[i].y - nowPos.y)) <
            min_dist)
        {
            min_dist =
                sqrt((targetPos[i].x - nowPos.x) * (targetPos[i].x - nowPos.x) +
                     (targetPos[i].y - nowPos.y) * (targetPos[i].y - nowPos.y));
            idx = i;
        }
    }
    return idx;
}

// Function to check if a point lies within the extended region of a line
bool check_path(const Point2D &path_point, const Point2D &obs_point)
{
    double dist = sqrt((obs_point.x - path_point.x) * (obs_point.x - path_point.x) + (obs_point.y - path_point.y) * (obs_point.y - path_point.y));
    return (dist > obs_size); // Check if the distance is within the extended region
}

class ListenerNode : public rclcpp::Node
{
public:
    ListenerNode() : Node("listener_node")
    {
        // Create subscribers for two topics
        o1_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("o1_data", 10, std::bind(&ListenerNode::o1_callback, this, std::placeholders::_1));
        o2_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("o2_data", 10, std::bind(&ListenerNode::o2_callback, this, std::placeholders::_1));
        o3_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("o3_data", 10, std::bind(&ListenerNode::o3_callback, this, std::placeholders::_1));
        o4_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("o4_data", 10, std::bind(&ListenerNode::o4_callback, this, std::placeholders::_1));
        o5_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("o5_data", 10, std::bind(&ListenerNode::o5_callback, this, std::placeholders::_1));
        b1_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("b1_data", 10, std::bind(&ListenerNode::b1_callback, this, std::placeholders::_1));
        b2_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("b2_data", 10, std::bind(&ListenerNode::b2_callback, this, std::placeholders::_1));
        b3_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("b3_data", 10, std::bind(&ListenerNode::b3_callback, this, std::placeholders::_1));
        b4_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("b4_data", 10, std::bind(&ListenerNode::b4_callback, this, std::placeholders::_1));
        b5_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("b5_data", 10, std::bind(&ListenerNode::b5_callback, this, std::placeholders::_1));
        ball_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("ball_data", 10, std::bind(&ListenerNode::ball_callback, this, std::placeholders::_1));
        decision_target_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>("/o1/decision_target_data", 10, std::bind(&ListenerNode::decision_target_callback, this, std::placeholders::_1));
        target_array_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/o1/target_pos", 10);
    }

    // Callback funtions
    void o1_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        nowPos = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        my_pose_received = true;
        listening();
    }
    void o2_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[0] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot1_pose_received = true;
    }
    void o3_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[1] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot2_pose_received = true;
    }
    void o4_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[2] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot3_pose_received = true;
    }
    void o5_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[3] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot4_pose_received = true;
    }
    void b1_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[4] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot5_pose_received = true;
    }
    void b2_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[5] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot6_pose_received = true;
    }
    void b3_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[6] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot7_pose_received = true;
    }
    void b4_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[7] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot8_pose_received = true;
    }
    void b5_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        obs[8] = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        bot9_pose_received = true;
    }
    void ball_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        ballPos = Point2D(msg->data[0], msg->data[1], 0);
        ball_pose_received = true;
    }
    void decision_target_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        cout << "New decision terget data received" << endl;
        finalPos = Point2D(msg->data[0], msg->data[1], msg->data[2]);
        target_pose_received = true;
    }

    void listening()
    {
        if (!my_pose_received || !bot1_pose_received || !bot2_pose_received || !bot2_pose_received || !bot3_pose_received || !bot4_pose_received || !bot5_pose_received || !bot6_pose_received || !bot7_pose_received || !bot8_pose_received || !bot9_pose_received || !ball_pose_received || !target_pose_received)
            return;

        /*for (const auto &bot_pose : obs)
        {
            if (finalPos.x == bot_pose.x && finalPos.y == bot_pose.y)
            {
                std::cerr << "Target pose conflicts with an existing bot's pose. Shifting target randomly." << std::endl;

                // Generate random r and theta within specified ranges
                double min_r = 0.5, max_r = 2.0; // Minimum and maximum radius for shifting
                double min_theta = 0.0, max_theta = 2 * M_PI; // Full circular range for theta

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dist_r(min_r, max_r);
                std::uniform_real_distribution<> dist_theta(min_theta, max_theta);

                double r = dist_r(gen);
                double theta = dist_theta(gen);

                // Shift the target pose
                finalPos.x += r * cos(theta);
                finalPos.y += r * sin(theta);

                std::cout << "New target position: (" << finalPos.x << ", " << finalPos.y << ")" << std::endl;
                break; // Exit the loop as we've shifted the target
            }
        }*/

        // check if isok is true for all points in planned path
        for (int i = 0; i < targetPos.size() && flag == 0; i++)
        {
            for (auto curr_obs : obs)
            {
                if (!check_path(targetPos[i], curr_obs))
                {
                    flag = 1;
                    break;
                }
            }
        }

        /*for (int i = 0; i < 9; i++ ) {
            double dist_from_target = sqrt((obs[i].x - finalPos.x)*(obs[i].x - finalPos.x) + (obs[i].y - finalPos.y)*(obs[i].y - finalPos.y));
            if (dist_from_target < obs_size) {
                targetPos.clear();
                targetPos.push_back(nowPos);
                iteration_count++;
                window->visualizeGame(targetPos, nowPos, findclosestpoint(targetPos, nowPos), nowPos.theta, obs, ballPos);
                publish_next_point(targetPos, nowPos);
                return;
            }
        }*/

        if (flag || iteration_count == 0 || (targetPos.size() > 0 && (finalPos.x != targetPos.back().x || finalPos.y != targetPos.back().y)))
        {
            try
            {
                targetPos.clear();
                targetPos = plan(runTime, A, B, obs, plannerType, objectiveType, nowPos, finalPos, ballPos);
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                std::cout << "Could not find path" << std::endl;

                targetPos.clear();
                targetPos.push_back(nowPos);

                /*double min_r = 0.5, max_r = 2.0;
                double min_theta = 0.0, max_theta = 2 * M_PI;

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dist_r(min_r, max_r);
                std::uniform_real_distribution<> dist_theta(min_theta, max_theta);

                double r = dist_r(gen);
                double theta = dist_theta(gen);

                finalPos.x += r * cos(theta);
                finalPos.y += r * sin(theta);

                std::cout << "New target position: (" << finalPos.x << ", " << finalPos.y << ")" << std::endl;
                targetPos.push_back(nowPos);
                targetPos.push_back(finalPos);*/
            }

            flag = 0;
        }

        if (sqrt((targetPos.back().x - finalPos.x) * (targetPos.back().x - finalPos.x) + (targetPos.back().y - finalPos.y) * (targetPos.back().y - finalPos.y)) > 0.1)
        {
            targetPos.clear();
            targetPos.push_back(nowPos);
        }

        iteration_count++;
        window->visualizeGame(targetPos, nowPos, findclosestpoint(targetPos, nowPos), nowPos.theta, obs, ballPos);
        publish_next_point(targetPos, nowPos);
    }

    // Variables to store received messages
    bool my_pose_received{false}, bot1_pose_received{false}, bot2_pose_received{false}, bot3_pose_received{false}, bot4_pose_received{false}, bot5_pose_received{false}, bot6_pose_received{false}, bot7_pose_received{false}, bot8_pose_received{false}, bot9_pose_received{false}, ball_pose_received{false}, target_pose_received{false};

    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr o1_subscriber, o2_subscriber, o3_subscriber, o4_subscriber, o5_subscriber, b1_subscriber, b2_subscriber, b3_subscriber, b4_subscriber, b5_subscriber, ball_subscriber, decision_target_subscriber;

    // Publisher
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr target_array_publisher;

    void publish_next_point(std::vector<Point2D> &path, Point2D &nowPos)
    {
        if (!path.empty())
        {
            int idx = findclosestpoint(path, nowPos);
        
            auto prev_point = path[min(idx,path.size()-2)];
            auto next_point = path[min(idx + 1, path.size() - 1)];
            auto next_to_next_point = path[min(idx + 2, path.size() - 1)];

            double x0 = prev_point.x;
            double y0 = prev_point.y;

            double x1 = next_point.x;
            double y1 = next_point.y;

            double x2 = next_to_next_point.x;
            double y2 = next_to_next_point.y;

            double angle = atan2((y2 - y1), (x2 - x1));

            if (idx + 2 == path.size() - 1 && idx + 1 != path.size() - 1)
            {
                double angle = atan2((y1 - path[idx].y), (x1 - path[idx].x));
            }

            double dist = sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
            
            // double c = 2.0, lambda = 0.3;
            // double speed = c * (dist - lambda);
            double speed = 0;
            idx++;
            if (idx > 0 && idx < path.size())
                {
                    // Points: previous, current, and next
                    Point2D p_prev = path[idx - 1];
                    Point2D p_curr = path[idx];
                    Point2D p_next = path[idx + 1];

                    // Calculate distances along the path
                    double d1 = sqrt((p_curr.x - p_prev.x) * (p_curr.x - p_prev.x) + (p_curr.y - p_prev.y) * (p_curr.y - p_prev.y));
                    double d2 = sqrt((p_next.x - p_curr.x) * (p_next.x - p_curr.x) + (p_next.y - p_curr.y) * (p_next.y - p_curr.y));

                    // Calculate angles of the segments
                    double angle1 = atan2(p_curr.y - p_prev.y, p_curr.x - p_prev.x);
                    double angle2 = atan2(p_next.y - p_curr.y, p_next.x - p_curr.x);

                    // Difference in angle (make sure it is between -pi and pi)
                    double dtheta = fabs(set_angle(angle2 - angle1));

                    // To avoid division by zero, check if dtheta is very small
                    if (dtheta > 1e-6)
                    {
                        // Estimate the radius of curvature
                        double radius = (d1 + d2) / dtheta;
                        // Calculate maximum allowed speed for this curvature: v_max = sqrt(a_max * r)
                        speed = sqrt(max_acceleration * radius);
                        speed = min(max_speed,speed);
                    }
                    else
                    {
                        // If the path is nearly straight, curvature_speed remains high
                        speed = max_speed; // or set to a high value if desired
                    }
                }
            double vx = speed * cos(angle);
            double vy = speed * sin(angle);

            double bot_global_angle = set_angle(nowPos.theta);
            double ball_wrto_bot = set_angle(atan2((ballPos.y - nowPos.y), (ballPos.x - nowPos.x)));
            double rel_angle = set_angle(ball_wrto_bot - bot_global_angle);

            rel_angle = min(rel_angle, max_angle);
            rel_angle = max(rel_angle, -max_angle);

            // double theta = rel_angle + bot_global_angle;
            double theta = finalPos.theta;

            
            double omega = 0;

            cout << "Next Point: " << x1 << " " << y1 << " " << theta << endl;
            cout << "Next velocity: " << vx << " " << vy << " " << omega << " " << speed << endl;

            std_msgs::msg::Float32MultiArray message;
            message.data = {x1, y1, theta, vx, vy, omega, speed, idx, path.size(),x0, y0};
            target_array_publisher->publish(message);
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ListenerNode>());
    rclcpp::shutdown();
    return 0;
}

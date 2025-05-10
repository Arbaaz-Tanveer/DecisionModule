#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/twist.hpp"

USING_NAMESPACE_ACADO

double current_x = 0.0, target_x =0, next_x = 0.0;
double current_y = 0.0, target_y = 0, next_y = 0.0;
double current_theta = 0.0, target_theta = 0.0, next_theta = 0.0;
double current_vx = 0.0, target_vx = 0.0, next_vx = 0.0;
double current_vy = 0.0, target_vy = 0.0, next_vy = 0.0;
double current_omega = 0.0, target_omega = 0.0, next_omega = 0.0;
double speed = 0.0;
double idx = 0.0;
double path_size = 0.0;


//new variables 
double max_omega = 1;


// Differential States and Controls
DifferentialState x, y, theta, vx, vy, omega;
Control ax, ay, alpha;
Parameter T;

// Parameters for robot model
double R = 0.05;
double d = 0.5;
double dt = 2.0; // time step for each iteration

DifferentialEquation f(0.0, T);

class MyRobotNode : public rclcpp::Node
{
public:
    MyRobotNode() : Node("my_robot_node")
    {
        // Publisher for cmd_vel (Twist)
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/o3/cmd_vel", 10);

        // Subscriber 1 for Float32MultiArray
        o1_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "o3_data", 10, std::bind(&MyRobotNode::callback_1, this, std::placeholders::_1));

        // Subscriber 2 for Float32MultiArray
        target_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/o3/target_pos", 10, std::bind(&MyRobotNode::callback_2, this, std::placeholders::_1));

	f << dot(x) == vx;
	f << dot(y) == vy;
	f << dot(theta) == omega;
	f << dot(vx) == ax;
	f << dot(vy) == ay;
	f << dot(omega) == alpha;
    }

    // Callback for o1_data from simulation
    void callback_1(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
	current_x = msg->data[0];
	current_y = msg->data[1];
	current_theta = msg->data[2];
	
	current_vx = next_vx;
	current_vy = next_vy;
	// current_omega = next_omega;

	// Set up the OCP for one iteration
        OCP ocp(0.0, T);
        ocp.minimizeMayerTerm(T);
        ocp.subjectTo(f);

        // State and control constraints
        ocp.subjectTo(-2 <= vx <= 2);
        ocp.subjectTo(-2 <= vy <= 2);
        // ocp.subjectTo(-2.0 <= omega <= 2.0);  // for python
        ocp.subjectTo(-2.0 <= ax <= 2.0);
        ocp.subjectTo(-2.0 <= ay <= 2.0);
        // ocp.subjectTo(-1.5 <= alpha <= 1.5);
        ocp.subjectTo(0.1 <= T <= 4.0); // Relaxed T bounds

        // Initial conditions
        ocp.subjectTo(AT_START, x == current_x);
        ocp.subjectTo(AT_START, y == current_y);
        // ocp.subjectTo(AT_START, theta == current_theta);
        ocp.subjectTo(AT_START, vx == current_vx);
        ocp.subjectTo(AT_START, vy == current_vy);
        // ocp.subjectTo(AT_START, omega == current_omega);
	
	std::cout << "Current State -> X: " << current_x << ", Y: " << current_y
                  << ", Theta: " << current_theta << ", Vx: " << current_vx
                  << ", Vy: " << current_vy << ", Omega: " << current_omega << std::endl;

        // Final state constraints towards target (moving only in x-direction)
        ocp.subjectTo(AT_END, x == target_x);   // Target position in x
        ocp.subjectTo(AT_END, y == target_y);       // Stop at the target
        // ocp.subjectTo(AT_END, theta == target_theta);
	ocp.subjectTo(AT_END, vx == 0);
        ocp.subjectTo(AT_END, vy == 0);
        // ocp.subjectTo(AT_END, omega == 0);
	
	std::cout << "Target State -> X: " << target_x << ", Y: " << target_y
                  << ", Theta: " << target_theta << ", Vx: " << target_vx
                  << ", Vy: " << target_vy << ", Omega: " << target_omega << std::endl;

	// Solver setup
        OptimizationAlgorithm algorithm(ocp);
        algorithm.set(PRINTLEVEL, LOW);
	if (algorithm.solve() == SUCCESSFUL_RETURN) {
		std::cout << "Algorithm successfully solved" << std::endl;
	}
	else {
		std::cout << "Algorithm not solved" << std::endl;
		return;
	}
        
	// Extract solution
        VariablesGrid states;
        algorithm.getDifferentialStates(states);

	// Update the state variables with the new state from the solution
        DVector next_state = states.getVector(1);  // Get state at next time step
        next_x = next_state(0);
        next_y = next_state(1);
        next_theta = next_state(2);
        next_vx = next_state(3);
        next_vy = next_state(4);
	// next_omega = next_state(5);
        if(target_theta > current_theta){
                next_omega = max_omega;
        }else{
                next_omega = -max_omega;
        }


	if (speed < 0.4) {
		next_vx *= (5.0/6.0);
		next_vy *= (5.0/6.0);
	} else if (speed < 0.5) {
		next_vx *= (5.5/6.0);
		next_vy *= (5.5/6.0);
	}
	
	
	if (idx >= path_size - 1) {
		next_vx /= 1.5;
		next_vy /= 1.5;
	} else if (idx >= path_size - 2) {
		next_vx /= 1.2;
		next_vx /= 1.2;
	}
	else if (idx >= path_size - 3) {
		next_vx /= 1.1;
		next_vx /= 1.1;
	}

	// Log the state
        std::cout << "Next State -> X: " << next_x << ", Y: " << next_y
                  << ", Theta: " << next_theta << ", Vx: " << next_vx
                  << ", Vy: " << next_vy << ", Omega: " << next_omega << std::endl;

        
        

        if (abs(target_theta - current_theta) < 0.05) {
                next_omega = 0;
        } else if (abs(target_theta - current_theta) < 0.3) {
               
                next_omega /= 8;
        }
        // std::cout<<"Bot stopped, next_omega = "<< next_omega <<std::endl;
        	
    	
	
	// Publish the Twist message
	geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x = next_vx;
	twist_msg.linear.y = next_vy;
        twist_msg.angular.z = next_omega;
        publisher_->publish(twist_msg);
    }

    // Callback for subscriber 2
    void callback_2(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
	target_x = msg->data[0];
	target_y = msg->data[1];
	target_theta = msg->data[2];
	target_vx = msg->data[3];
	target_vy = msg->data[4];
	target_omega = msg->data[5];
	speed = msg->data[6];
        idx = msg->data[7];
        path_size = msg->data[8];
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr o1_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr target_subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyRobotNode>());
    rclcpp::shutdown();
    return 0;
}


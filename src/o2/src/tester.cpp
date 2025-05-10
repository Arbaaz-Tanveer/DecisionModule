#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <iostream>

USING_NAMESPACE_ACADO

int main() {
    // Differential States and Controls
    DifferentialState x, y, theta, vx, vy, omega;
    Control ax, ay, alpha;
    Parameter T;

    // Parameters for robot model
    const double R = 0.05;
    const double d = 0.1;
    const double dt = 2.0; // time step for each iteration

    // Initial states
    double current_x = 0.0, current_y = 0.0, current_theta = 0.0;
    double current_vx = 0.0, current_vy = 0.0, current_omega = 0.0;

    // Differential Equation
    DifferentialEquation f(0.0, T);
    f << dot(x) == vx;
    f << dot(y) == vy;
    f << dot(theta) == omega;
    f << dot(vx) == ax;
    f << dot(vy) == ay;
    f << dot(omega) == alpha;

    while (1) {
        // Set up the OCP for one iteration
        OCP ocp(0.0, T);
        ocp.minimizeMayerTerm(T);
        ocp.subjectTo(f);
        
        // State and control constraints
        // ocp.subjectTo(-5.0 <= vx <= 5.0);
        // ocp.subjectTo(-5.0 <= vy <= 5.0);
        // ocp.subjectTo(-1.0 <= omega <= 1.0);
        // ocp.subjectTo(-2.0 <= ax <= 2.0);
        // ocp.subjectTo(-2.0 <= ay <= 2.0);
        // ocp.subjectTo(-1.0 <= alpha <= 1.0);
        // ocp.subjectTo(0.01 <= T <= 50.0); // Relaxed T bounds
	
	ocp.subjectTo(-1.0 <= vx <= 1.0);     // Updated velocity constraints
        ocp.subjectTo(-1.0 <= vy <= 1.0);     // Allow reasonable velocity in y
        ocp.subjectTo(-0.5 <= omega <= 0.5);  // Angular velocity constraints
        ocp.subjectTo(-1.0 <= ax <= 1.0);     // Acceleration limits
        ocp.subjectTo(-1.0 <= ay <= 1.0);     // Acceleration in y
        ocp.subjectTo(-0.5 <= alpha <= 0.5);  // Angular acceleration
        ocp.subjectTo(0.1 <= T <= 4.0);      // Relaxed time horizon	
						    
        // Initial conditions
        ocp.subjectTo(AT_START, x == current_x);
        ocp.subjectTo(AT_START, y == current_y);
        ocp.subjectTo(AT_START, theta == current_theta);
        ocp.subjectTo(AT_START, vx == current_vx);
        ocp.subjectTo(AT_START, vy == current_vy);
        ocp.subjectTo(AT_START, omega == current_omega);

        // Final state constraints towards target (moving only in x-direction)
        ocp.subjectTo(AT_END, x == 0.5);   // Target position in x
        ocp.subjectTo(AT_END, vx == 0.0);       // Stop at the target
        ocp.subjectTo(AT_END, y == 0.0);
        ocp.subjectTo(AT_END, theta == 0.0);
        ocp.subjectTo(AT_END, vy == 0.0);
        ocp.subjectTo(AT_END, omega == 0.0);

        // Solver setup 
        OptimizationAlgorithm algorithm(ocp);
        algorithm.set(PRINTLEVEL, LOW);

        // Solve the optimization problem and check the success status
        if (algorithm.solve() != SUCCESSFUL_RETURN) {
            std::cout << "Solver failed to find a feasible solution.\n";
            break;
        }

        // Extract solution
        VariablesGrid states;
        algorithm.getDifferentialStates(states);
        
        // Update the state variables with the new state from the solution
        DVector next_state = states.getVector(1);  // Get state at next time step
        current_x = next_state(0);
        current_y = next_state(1);
        current_theta = next_state(2);
        current_vx = next_state(3);
        current_vy = next_state(4);
        current_omega = next_state(5);

        // Log the state
        std::cout << "Current State -> X: " << current_x << ", Y: " << current_y
                  << ", Theta: " << current_theta << ", Vx: " << current_vx
                  << ", Vy: " << current_vy << ", Omega: " << current_omega << std::endl;
	std::cout << "Next State -> X: " << next_state(0) << ", Y: " << next_state(1)
                  << ", Theta: " << next_state(2) << ", Vx: " << next_state(3)
                  << ", Vy: " << next_state(4) << ", Omega: " << next_state(5) << std::endl;

    }

    return 0;
}

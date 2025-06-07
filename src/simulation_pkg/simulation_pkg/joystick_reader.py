import pygame
import time

# Initialize pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Check for joystick
if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
    exit()

# Connect to the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick connected: {joystick.get_name()}")
print(f"Number of axes: {joystick.get_numaxes()}")
print(f"Number of buttons: {joystick.get_numbuttons()}")
print(f"Number of hats: {joystick.get_numhats()}")

try:
    while True:
        pygame.event.pump()  # Process event queue

        # Read axes (analog sticks or triggers)
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]

        # Read buttons (pressed/released)
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        # Read hats (D-pad)
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

        print("Axes:", axes)
        print("Buttons:", buttons)
        print("Hats:", hats)
        print("-----")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    joystick.quit()
    pygame.quit()
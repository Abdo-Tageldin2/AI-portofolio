import pygame
import sys

# Run Value Iteration
ROWS, COLS = 6, 6  # Number of rows and columns in the grid
states = [(i, j) for i in range(ROWS) for j in range(COLS)]  # List of all possible states (cell positions) in the grid
actions = ["UP", "DOWN", "LEFT", "RIGHT"]  # Possible actions (movements) the agent can take

# Variables for start, goal, and obstacles
goal_state = (4, 4)  # The cell where the agent should reach to gain a positive reward
obstacles = [(2, 2), (1, 3), (3, 1)]  # Cells that the agent cannot move onto
start_state = None  # The initial position of the agent, to be selected by the user

path = []  # Stores the sequence of states representing the optimal path from start to goal

# Pygame setup
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 500, 500  # The width and height of the game window in pixels
CELL_SIZE = WIDTH // COLS  # The size of each individual cell (square) in the grid

# Colors (in RGB format)
WHITE = (255, 255, 255)        # Background/empty cell color
BLACK = (0, 0, 0)              # Grid line and text color
GOAL_COLOR = (0, 255, 0)       # Color for the goal state cell
OBSTACLE_COLOR = (128, 128, 128)  # Color for obstacle cells
PATH_COLOR = (0, 0, 255)       # Color for cells on the computed optimal path
START_COLOR = (255, 165, 0)    # Color for the selected start state

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create the Pygame window of specified size
pygame.display.set_caption("MDP Value Iteration Visualization")  # Set the window title

# Fonts
font = pygame.font.Font(None, 36)  # Font for displaying the values of states in the cells

def reward(state):
    # Define the reward function.
    # The agent receives a positive reward if it reaches the goal state, and a negative reward otherwise.
    return 10 if state == goal_state else -1

def transition(state, action):
    # Defines the next state given the current state and an action.
    # The agent moves one cell in the specified direction if possible.
    # If the move is blocked by an obstacle or out of the grid bounds, the state does not change.
    if state in obstacles:
        return state  # No movement allowed if currently on an obstacle (though ideally agent shouldn't be there)
    x, y = state
    # Check the action and move accordingly if valid
    if action == "UP" and x > 0 and (x - 1, y) not in obstacles:
        return (x - 1, y)
    if action == "DOWN" and x < ROWS - 1 and (x + 1, y) not in obstacles:
        return (x + 1, y)
    if action == "LEFT" and y > 0 and (x, y - 1) not in obstacles:
        return (x, y - 1)
    if action == "RIGHT" and y < COLS - 1 and (x, y + 1) not in obstacles:
        return (x, y + 1)
    # If no valid move, remain in the same state
    return state

def value_iteration(states, actions, theta=1e-4, gamma=0.9):
    # Perform value iteration to compute the value function for each state.
    # The value function V(s) represents the maximum expected reward achievable from state s.
    # gamma is the discount factor, theta is the threshold for stopping.
    V = {state: 0 for state in states}  # Initialize all state values to 0
    while True:
        delta = 0  # Tracks the maximum difference between new and old values in this iteration
        for state in states:
            if state in obstacles:
                # Skip updating values for obstacles as the agent shouldn't move onto them
                continue
            v = V[state]  # Current value of the state
            # Compute the new value as the maximum over all actions of [reward(s) + gamma * V(s')]
            # where s' is the next state after taking that action
            V[state] = max(reward(state) + gamma * V[transition(state, action)] for action in actions)
            delta = max(delta, abs(v - V[state]))  # Update delta with the change in value
        # Stop iterating once values have converged (change is less than theta)
        if delta < theta:
            break
    return V

def find_optimal_path(start_state, values):
    # Given the computed value function, find the optimal path from start_state to goal_state.
    # The path is obtained by repeatedly choosing the action that leads to the state with the highest value.
    current_state = start_state
    path.clear()
    path.append(current_state)
    while current_state != goal_state:
        # Choose the action that leads to the highest value in the next state
        best_action = max(actions, key=lambda action: values[transition(current_state, action)])
        next_state = transition(current_state, best_action)
        if next_state == current_state:
            # No further movement possible; break out if stuck
            break
        path.append(next_state)
        current_state = next_state

def draw_grid(values):
    # Draw the entire grid, including:
    # - Obstacles
    # - Goal state
    # - Start state
    # - Optimal path
    # - Grid lines
    # Also display the value of each state in its cell.
    for i in range(ROWS):
        for j in range(COLS):
            x = j * CELL_SIZE
            y = i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            # Determine the cell color based on its role
            if (i, j) in obstacles:
                pygame.draw.rect(screen, OBSTACLE_COLOR, rect)
            elif (i, j) == goal_state:
                pygame.draw.rect(screen, GOAL_COLOR, rect)
            elif (i, j) == start_state:
                pygame.draw.rect(screen, START_COLOR, rect)
            elif (i, j) in path:
                pygame.draw.rect(screen, PATH_COLOR, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            # Draw a black outline for each cell
            pygame.draw.rect(screen, BLACK, rect, 1)

            # Display the value of this state, if known and not an obstacle
            if (i, j) not in obstacles and values:
                value_text = font.render(f"{values[(i, j)]:.2f}", True, BLACK)
                text_rect = value_text.get_rect(center=rect.center)
                screen.blit(value_text, text_rect)

def main():
    # The main function handles the event loop and updates the environment based on user input.
    # It waits for the user to select a start state, computes values, and then shows the optimal path.
    global start_state, goal_state, obstacles

    values = None  # Will hold the value function once computed
    selecting_start = True  # Flag to indicate user is selecting the initial start state

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the window is closed, exit the program
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and selecting_start:
                # If user clicks while selecting the start state, set it if valid
                x, y = event.pos
                row, col = y // CELL_SIZE, x // CELL_SIZE
                # Ensure the start is not on an obstacle or the goal
                if (row, col) not in obstacles and (row, col) != goal_state:
                    start_state = (row, col)
                    selecting_start = False
                    # Once start is selected, run value iteration and find the path
                    values = value_iteration(states, actions)
                    find_optimal_path(start_state, values)

            elif event.type == pygame.MOUSEBUTTONDOWN and not selecting_start:
                # Once start is chosen, left and right clicks can modify the environment
                x, y = event.pos
                row, col = y // CELL_SIZE, x // CELL_SIZE
                if event.button == 1:  # Left click
                    # Left click changes obstacles or resets the start state
                    if (row, col) == goal_state:
                        # Don't allow changing the goal cell
                        continue
                    elif (row, col) == start_state:
                        # Reset start state
                        start_state = None
                        selecting_start = True
                    elif (row, col) in obstacles:
                        # Remove an obstacle if it was clicked
                        obstacles.remove((row, col))
                    elif start_state is None:
                        # If no start is chosen, set this cell as start
                        start_state = (row, col)
                    elif (row, col) not in obstacles:
                        # Otherwise, add a new obstacle
                        obstacles.append((row, col))
                elif event.button == 3:  # Right click
                    # Right click can change the goal state if the cell is valid
                    if (row, col) not in obstacles and (row, col) != start_state:
                        goal_state = (row, col)

                # After modifying the environment, recalculate values and path if start is chosen
                if not selecting_start:
                    values = value_iteration(states, actions)
                    find_optimal_path(start_state, values)

        # Clear the screen
        screen.fill(WHITE)
        # Draw the current state of the grid and values
        draw_grid(values)
        # Update the display
        pygame.display.flip()

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Define possible movement actions (Up, Down, Left, Right)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_LIST = list(ACTIONS.keys())


def value_iteration(grid, size, gamma=0.9, theta=1e-4):
    """
    Perform Value Iteration Algorithm to find the optimal policy.
    :param grid: The grid representation with "start", "goal", "obstacle".
    :param size: The size of the grid (n x n).
    :param gamma: Discount factor for future rewards.
    :param theta: Convergence threshold for value updates.
    :return: Value matrix V(s), Policy matrix π(s), and the optimal path from start to goal.
    """
    V = np.zeros((size, size))  # Initialize Value Function
    # Initialize policy with empty values
    policy = np.full((size, size), " ", dtype='<U5')

    start, goal = None, None
    for i in range(size):
        for j in range(size):
            if grid[i][j] == "start":
                start = (i, j)
            elif grid[i][j] == "goal":
                goal = (i, j)

    if not start or not goal:
        return None, None, None  # If no start/goal is set, return None

    while True:
        delta = 0
        new_V = np.copy(V)  # Create a copy of the value function

        for i in range(size):
            for j in range(size):
                if (i, j) == goal:  # Goal state has fixed value
                    continue
                if grid[i][j] == "obstacle":  # Obstacles remain -1
                    V[i, j] = -1
                    continue

                best_value = float('-inf')
                best_action = None

                # Iterate over all possible actions (Up, Down, Left, Right)
                for action in ACTION_LIST:
                    di, dj = ACTIONS[action]
                    ni, nj = i + di, j + dj

                    # Ensure the new position is within bounds and not an obstacle
                    if 0 <= ni < size and 0 <= nj < size and grid[ni][nj] != "obstacle":
                        reward = -1  # Every move has a small cost
                        new_value = reward + gamma * V[ni, nj]

                        if new_value > best_value:
                            best_value = new_value
                            best_action = action

                if best_action:
                    new_V[i, j] = best_value  # Update value function
                    policy[i, j] = best_action  # Update policy

                # Check for convergence
                delta = max(delta, abs(V[i, j] - new_V[i, j]))

        V = new_V
        if delta < theta:  # Stop if the value function has converged
            break

    # Compute the best path from start to goal
    path = []
    position = start
    while position != goal:
        path.append(position)
        action = policy[position]
        if action == " ":
            break  # Stop if no valid action is found
        di, dj = ACTIONS[action]
        position = (position[0] + di, position[1] + dj)
        if position in path:
            break  # Avoid infinite loops

    # Return value function, policy, and computed path
    return V.tolist(), policy.tolist(), path


@app.route('/')
def index():
    """ Render the main webpage """
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Receive the grid configuration from the frontend, run Value Iteration,
    and return the computed Value Function and Policy.
    """
    data = request.json
    grid = data["grid"]
    size = data["size"]

    V, policy, path = value_iteration(grid, size)
    if V is None:
        return jsonify({"error": "Please ensure start and goal positions are set"})

    return jsonify({"values": V, "policy": policy, "path": path})


if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask server


from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_LIST = list(ACTIONS.keys())


def value_iteration(grid, size, gamma=0.9, theta=1e-4):
    V = np.zeros((size, size))
    policy = np.full((size, size), " ", dtype='<U5')

    start, goal = None, None
    for i in range(size):
        for j in range(size):
            if grid[i][j] == "start":
                start = (i, j)
            elif grid[i][j] == "goal":
                goal = (i, j)

    if not start or not goal:
        return None, None, None

    traces = []

    for iteration in range(50):  # 限制最大迭代次數
        delta = 0
        new_V = np.copy(V)

        for i in range(size):
            for j in range(size):
                if (i, j) == goal:
                    continue
                if grid[i][j] == "obstacle":
                    V[i, j] = -1
                    continue

                best_value = float('-inf')
                best_action = None

                for action in ACTION_LIST:
                    di, dj = ACTIONS[action]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < size and 0 <= nj < size and grid[ni][nj] != "obstacle":
                        reward = -1
                        new_value = reward + gamma * V[ni, nj]
                        if new_value > best_value:
                            best_value = new_value
                            best_action = action

                if best_action:
                    new_V[i, j] = best_value
                    policy[i, j] = best_action

                delta = max(delta, abs(V[i, j] - new_V[i, j]))

        V = new_V
        traces.append(generate_trace(
            policy, grid, start, goal, size, epsilon=0.2))

        if delta < theta:
            break

    return V.tolist(), policy.tolist(), traces


def generate_trace(policy, grid, start, goal, size, epsilon=0.2):
    trace = []
    pos = start
    visited = set()

    for _ in range(size * size):
        trace.append(pos)
        if pos == goal or pos in visited:
            break
        visited.add(pos)

        if random.random() < epsilon:
            action = random.choice(ACTION_LIST)
        else:
            action = policy[pos[0]][pos[1]]

        if action == " ":
            break

        di, dj = ACTIONS[action]
        new_pos = (pos[0] + di, pos[1] + dj)

        if 0 <= new_pos[0] < size and 0 <= new_pos[1] < size:
            if grid[new_pos[0]][new_pos[1]] != "obstacle":
                pos = new_pos
            else:
                break
        else:
            break

    return trace


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    grid = data["grid"]
    size = data["size"]

    V, policy, traces = value_iteration(grid, size)
    if V is None:
        return jsonify({"error": "Please ensure start and goal positions are set"})

    return jsonify({"values": V, "policy": policy, "traces": traces})


if __name__ == '__main__':
    app.run(debug=True)

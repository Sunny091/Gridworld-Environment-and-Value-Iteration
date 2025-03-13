from flask import Flask, render_template, request, jsonify
import numpy as np

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

    while True:
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
        if delta < theta:
            break

    path = []
    position = start
    while position != goal:
        path.append(position)
        action = policy[position]
        if action == " ":
            break
        di, dj = ACTIONS[action]
        position = (position[0] + di, position[1] + dj)
        if position in path:
            break

    return V.tolist(), policy.tolist(), path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    grid = data["grid"]
    size = data["size"]

    V, policy, path = value_iteration(grid, size)
    if V is None:
        return jsonify({"error": "請確保起點和終點都已設定"})

    return jsonify({"values": V, "policy": policy, "path": path})


if __name__ == '__main__':
    app.run(debug=True)

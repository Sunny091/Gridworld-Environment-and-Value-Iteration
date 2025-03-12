from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

grids = {}
n = 5  # Default grid size

actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}


def generate_random_policy(n):
    return [[random.choice(list(actions.keys())) for _ in range(n)] for _ in range(n)]


def policy_evaluation(policy, n, gamma=0.9, theta=0.01):
    V = np.zeros((n, n))
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if (i, j) in grids and grids[(i, j)] == 'obstacle':
                    continue
                v = V[i, j]
                action = policy[i][j]
                ni, nj = i + actions[action][0], j + actions[action][1]
                if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in grids:
                    V[i, j] = -1 + gamma * V[ni, nj]
                else:
                    V[i, j] = -1 + gamma * V[i, j]
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V.tolist()


@app.route('/')
def index():
    return render_template('index.html', n=n)


@app.route('/set_grid', methods=['POST'])
def set_grid():
    global grids, n
    data = request.json
    n = data['n']
    grids = {}  # Reset grid
    return jsonify({'message': 'Grid size updated'})


@app.route('/update_grid', methods=['POST'])
def update_grid():
    global grids
    data = request.json
    x, y, cell_type = data['x'], data['y'], data['type']
    if cell_type == 'clear':
        grids.pop((x, y), None)
    else:
        grids[(x, y)] = cell_type
    return jsonify({'message': 'Grid updated'})


@app.route('/generate_policy', methods=['GET'])
def generate_policy():
    policy = generate_random_policy(n)
    values = policy_evaluation(policy, n)
    return jsonify({'policy': policy, 'values': values})


if __name__ == '__main__':
    app.run(debug=True)

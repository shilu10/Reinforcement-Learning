# 🚀 Reinforcement Learning & Deep RL Algorithms from Scratch

This repository offers a comprehensive collection of Reinforcement Learning (RL) and Deep Reinforcement Learning (Deep RL) algorithms, implemented from scratch using **Python** and **TensorFlow 2**. It serves as both a learning resource and a practical toolkit for understanding and experimenting with various RL techniques.

---

## 📌 Features

- **Classical RL Algorithms**: Q-Learning, SARSA, Monte Carlo methods, and Temporal Difference (TD) learning.
- **Deep RL Algorithms**: DQN, Double DQN, Dueling DQN, DDPG, TD3, PPO, and SAC.
- **Experience Replay Variants**: Includes Vanilla, Combined Experience Replay (CER), and Prioritized Experience Replay (PER).
- **Custom Environments**: Integrated with OpenAI Gym (e.g., LunarLander, Pong), with wrappers/utilities for extended control.
- **Training Utilities**: Tools for monitoring and visualization, including **TensorBoard** and optional **Telegram bot notifications**.

---

## 🧠 Algorithms Implemented

### 📘 Value-Based Methods
- Q-Learning
- SARSA
- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN

### 📙 Policy-Based Methods
- Policy Gradient
- Actor-Critic
- Advantage Actor-Critic (A2C)
- Asynchronous Advantage Actor-Critic (A3C)

### 📕 Actor-Critic Methods
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Trust Region Policy Optimization (TRPO)

### 📗 Others
- Monte Carlo Methods
- Temporal Difference (TD) Prediction
- Dynamic Programming

---

## 🗂️ Project Structure

```text
Reinforcement-Learning/
├── A2C_Continuous/
├── A2C_Discrete/
├── A3C_Continuous/
├── A3C_Discrete/
├── DDPG/
├── DQN/
├── Double_DQN/
├── Dueling_DDQN/
├── Dueling_DQN/
├── DynamicProgramming_RL/
├── Monte_Carlo/
├── PPO/
├── Policy_Gradient/
├── Q_Learning/
├── SAC/
├── SARSA/
├── TD3/
├── TD_Prediction/
├── TRPO/
├── assets/
├── images/
└── README.md
```
Each directory corresponds to a specific algorithm or group of related algorithms, including implementation code and training scripts.

---

## 🛠️ Installation & Setup

### ✅ Prerequisites
- Python 3.7+
- Git

### 📦 Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shilu10/Reinforcement-Learning.git
   cd Reinforcement-Learning
2. **Create and Activate a Virtual Environment**:
    ```bash
    python3 -m venv rl_env
    source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate
    ```
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Running Experiments
▶️ Example: Training DQN on LunarLander
```bash
cd DQN
python train.py
```
🧪 Example: Evaluating a Trained Agent
```bash
python eval.py
```
Each folder includes a train.py and (optionally) eval.py. Some scripts may include configurations or model checkpoints.

📊 Monitoring & Visualization
- TensorBoard: For training visualization (tensorboard --logdir logs/).
- Matplotlib: Reward and loss plotting after training.
- Telegram Bot (Optional): Send live updates from training to a Telegram chat (requires bot token and chat ID).

## 📚 References & Resources

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- OpenAI. (2018). *Spinning Up in Deep RL*. [https://spinningup.openai.com](https://spinningup.openai.com)
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533. [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)
- OpenAI Gym Documentation. [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- Lilian Weng. (2018). *Policy Gradient Algorithms*. [https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
- Deep Reinforcement Learning Course by David Silver. [https://www.davidsilver.uk/teaching/](https://www.davidsilver.uk/teaching/)



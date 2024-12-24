
# PettingZoo Pistonball Environment using 2 Balls with PPO Implementation

This project demonstrates the implementation of the Proximal Policy Optimization (PPO) algorithm on PettingZoo's Pistonball environment with multi-agent reinforcement learning, focusing on emergent behaviors through the addition of a second ball.

## 🚀 Exciting Highlights

This project explores the fascinating dynamics of **emergent behaviors** in multi-agent reinforcement learning. By introducing a second ball into the Pistonball environment, we observe how agents self-organize to demonstrate:
- **Division of labor**: Agents prioritize tasks dynamically.
- **Dynamic balance**: Coordinated actions to maintain control over both balls.
- **Adaptive behaviors**: Strategies evolve in real-time based on challenges.

👉 **Read the detailed analysis and insights on my Medium article:**  
[Exciting Discovery: AI Pistonball Games and Emergent Collective Behaviors](https://medium.com/@ryanchen_1890/exciting-discovery-ai-pistonball-games-and-emergent-collective-behaviors-f5ba8f9e71c7?sk=5c9b0edd811efd9a847624235f0f9bf8)

## Prerequisites

- `pyenv` (Python version manager)
- `pipenv` (Python dependency manager)
- Homebrew (for macOS users)

## Installation & Setup

1. Create and navigate to the project directory:
   ```bash
   mkdir pettingzoo_pistonball
   cd pettingzoo_pistonball
   ```

2. Check available Python versions:
   ```bash
   pyenv versions
   ```

3. Install Python 3.11.10 if not available:
   ```bash
   pyenv install 3.11.10
   ```

4. Set up a virtual environment:
   ```bash
   pipenv --python 3.11.10
   pipenv shell
   ```

5. Install required packages:
   ```bash
   # PettingZoo with necessary environments
   pip install 'pettingzoo[butterfly,atari,testing]>=1.24.0'
   pip install 'pettingzoo[all]'

   # Additional dependencies
   brew install cmake
   pipenv install SuperSuit>=3.9.0
   pipenv install tensorboard>=2.11.2
   pipenv install torch>=1.13.1
   ```

6. Verify the installation:
   ```bash
   pip show pettingzoo
   ```

## Project Files

- `test1.py`: PPO training implementation
- `test2.py`: Evaluation and visualization
- `pistonball.py`: Modified environment with two balls

## Key Features

- **Implementation of PPO algorithm**: Training intelligent agents.
- **Multi-agent reinforcement learning**: Observing interactions in a dynamic system.
- **Modified Pistonball environment**: Introduction of two balls to study emergent phenomena.
- **Visualization**: Heatmaps and trajectories highlight emergent behaviors.
- **Analysis**: In-depth exploration of adaptive strategies and cooperative behaviors.

## Usage

1. Train the model:
   ```bash
   python test1.py
   ```

2. Evaluate and visualize the results:
   ```bash
   python test2.py
   ```

## Reference Resources

- [PettingZoo GitHub Repository](https://github.com/Farama-Foundation/PettingZoo)
- [Pistonball Environment Details](https://pettingzoo.farama.org/environments/butterfly/pistonball/)
- [CleanRL Tutorial for PPO](https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/)

## Contributing

Feel free to submit issues and enhancement requests.

## License

[MIT License](LICENSE)

---

### 🌟 Don't Miss Out!
To dive deeper into the concept of emergent behaviors and their profound implications for AI research, check out my Medium post:  
[Exciting Discovery: AI Pistonball Games and Emergent Collective Behaviors](https://medium.com/@ryanchen_1890/exciting-discovery-ai-pistonball-games-and-emergent-collective-behaviors-f5ba8f9e71c7?sk=5c9b0edd811efd9a847624235f0f9bf8)

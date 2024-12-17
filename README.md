# PettingZoo Pistonball Environment with PPO Implementation

This project demonstrates the implementation of Proximal Policy Optimization (PPO) algorithm on PettingZoo's Pistonball environment with multi-agent reinforcement learning.

## Prerequisites

- pyenv (Python version manager)
- pipenv (Python dependency manager)
- Homebrew (for macOS users)

## Installation & Setup

1. Create and navigate to project directory:
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
pyenv install --list  # Check available versions
pyenv install 3.11.10
```

4. Set up virtual environment:
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
brew install cmake  # Required for some packages
pipenv install SuperSuit>=3.9.0
pipenv install tensorboard>=2.11.2
pipenv install torch>=1.13.1
```

6. Verify installation:
```bash
pip show pettingzoo
```

## Project Files

- `test1.py`: PPO training implementation
- `test2.py`: Evaluation and visualization
- `pistonball.py`: Modified environment with two balls

## Key Features

- Implementation of PPO algorithm
- Multi-agent reinforcement learning
- Modified Pistonball environment with two balls
- Visualization of agent behaviors
- Analysis of emergent behaviors

## Usage

1. Training the model:
```bash
python test1.py
```

2. Evaluating and visualizing the results:
```bash
python test2.py
```

## Reference

This project is based on the PettingZoo tutorial for implementing PPO:
[PettingZoo PPO Tutorial](https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/)

## Contributing

Feel free to submit issues and enhancement requests.

## License

[MIT License](LICENSE)

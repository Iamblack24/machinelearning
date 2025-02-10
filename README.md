# Self-Learning AI

## Overview
A modular, adaptive AI system designed to learn through interaction, with capabilities to acquire knowledge incrementally via text and speech interfaces.

## Features
- Incremental learning mechanism
- Knowledge representation and storage
- Language understanding and generation
- Adaptive learning strategies

## Setup
1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Core Components
- `learner.py`: Core learning algorithms
- `knowledge_base.py`: Knowledge storage and retrieval

## Usage
```python
from src.learner import SelfLearningAgent

agent = SelfLearningAgent()
agent.interact("2 + 3 = 5")  # Teach mathematical concepts
result = agent.query("What is 2 + 3?")
print(result)
```

## Contributing
Contributions welcome! Please read the contributing guidelines before starting.

## License
MIT License
# machinelearning

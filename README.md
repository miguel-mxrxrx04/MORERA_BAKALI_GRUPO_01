# Reddit Sentiment Analysis Project

## Overview
This project performs sentiment analysis on Reddit posts, analyzing user sentiment across different subreddits and topics.

## Project Structure
- `src/`: Source code files
- `data/`: Data directory for Reddit posts
- `requirements/`: Dependency files
    - `gensim_requirements.txt`: Dependencies for Gensim-related processes
    - `main_requirements.txt`: Dependencies for main analysis

## Setup and Installation

### For Gensim-related processes
```bash
python -m venv gensim_env
source gensim_env/bin/activate  # On Windows use: gensim_env\Scripts\activate
pip install -r requirements/gensim_requirements.txt
```

### For Main Analysis
```bash
python -m venv main_env
source main_env/bin/activate  # On Windows use: main_env\Scripts\activate
pip install -r requirements/main_requirements.txt
```

**Important**: Always ensure you're using the correct virtual environment when running different parts of the project:
- Use `gensim_env` for word embedding and topic modeling
- Use `main_env` for sentiment analysis and other processing tasks


## License
No license, only for academic purposes.

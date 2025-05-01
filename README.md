## Team 👏
- **Founder:** Ayaan Khan
- **Cofounders:** 
  - Safdar Khan
  - Aditya Kumar
- **Team Members:**
  - Vijay 
  - Raj
  - Kevin
  - commit 1

## Project Overview 🔝

SyookAI is built using modern AI frameworks and computer vision libraries to deliver intelligent solutions. The project utilizes both PyTorch and TensorFlow for deep learning capabilities, along with OpenCV for image processing tasks.

## Technical Requirements 👨‍💻

### Dependencies

```bash
opencv-python    # Computer vision tasks
ultralyntics     # YOLO implementation
argparse         # Command-line argument parsing
torch           # PyTorch deep learning framework
tensorflow      # TensorFlow deep learning framework
mediapipe       # Google's ML solutions for media processing
matplotlib      # Data visualization and plotting library
```

### Legal Restrictions

This software cannot be used for defaming purposes.

### Installation

#### Poetry Installation (Recommended)
1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/ayaankhan28/SyookAI.git
cd SyookAI
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Activate the Poetry virtual environment:
```bash
poetry shell
```

#### Docker Installation
1. Make sure you have Docker installed on your system. If not, download and install from [Docker's official website](https://www.docker.com/get-started).

2. Clone the repository:
```bash
git clone https://github.com/ayaankhan28/SyookAI.git
cd SyookAI
```

3. Build the Docker image:
```bash
docker build -t syookai .
```

4. Run the Docker container:
```bash
docker run -it --name syookai-container syookai
```

5. For development with volume mounting (optional):
```bash
docker run -it --name syookai-dev -v $(pwd):/app syookai
```

#### Standard Installation
1. Clone the repository:
```bash
git clone https://github.com/ayaankhan28/SyookAI.git
cd SyookAI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Developer Installation Steps

1. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt  # Contains additional testing and development packages
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Set up environment variables:
```bash
cp .env.example .env  # Copy example environment file
# Edit .env with your configurations
```

5. Run tests:
```bash
pytest
```

## Usage 🔐

The project contains multiple Python scripts for different functionalities:

- `inference.py` - Main inference script for model predictions

Please refer to individual script documentation for specific usage instructions.

## Future Plans 🏯

We have to build the worlds first robo glass company

## Contact ✼

For any queries or contributions, please reach out to the project maintainers or create an issue in the GitHub repository.

---
Made with ♼♽ at Brilliance Labs
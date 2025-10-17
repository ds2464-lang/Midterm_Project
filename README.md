Enhanced Calculator Command-Line Application

This project is an Enhanced Calculator Command-Line Application featuring:

Command-line interface (CLI)

Undo/redo functionality

History management

Logging of operations

1. Install and Configure Git
Install Git

MacOS (using Homebrew):

brew install git


Windows:
Download and install Git for Windows
. Accept default options during installation.

Verify Installation
git --version

Configure Git Globals

Set your name and email for commit tracking:

git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"


Verify settings:

git config --list

Generate SSH Keys and Connect to GitHub (one-time per machine)
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519


Copy your SSH public key:

Mac/Linux: cat ~/.ssh/id_ed25519.pub | pbcopy

Windows (Git Bash): cat ~/.ssh/id_ed25519.pub | clip

Add it to GitHub: Settings → SSH and GPG keys → New SSH Key.
Test connection:

ssh -T git@github.com

2. Clone the Repository
git clone https://github.com/ds2464-lang/Midterm_Project.git
cd Midterm_Project

3. Install Python 3.10+
Install Python

MacOS (Homebrew):

brew install python


Windows:
Download Python for Windows
.
Make sure to check Add Python to PATH during installation.

Verify Installation
python3 --version
# or
python --version

Create and Activate a Virtual Environment (Optional but Recommended)
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate.bat  # Windows

Install Required Packages
pip install -r requirements.txt

4. Running the Project
python main.py
# or
python3 main.py

5. Command-Line Interface (CLI)

After running python main.py, you will see an initial greeting.
The CLI will prompt you for:

Command

First number

Second number

Available commands:

help – Display information

history – View operation history

clear – Clear all history

undo – Undo most recent operation

redo – Redo most recently undone operation

exit – Exit the application

After exit, a farewell message will be displayed.

6. Running Tests
pytest
# or with coverage
pytest -cov

7. GitHub Actions Workflow

The GitHub Actions workflow automates:

Testing of Python code

Code quality checks (linting, formatting)

Dependency management

Automated deployment

Documentation and build checks

Integration with GitHub events

In this project, the workflow ensures automated testing and an overall coverage of 90% or higher.

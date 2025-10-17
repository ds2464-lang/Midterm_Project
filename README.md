Project: Enhanced Calculator Command-Line Application
The following project is an attempt at an Enhanced Calculator Command-Line Application.
It features a command-line interface, undo/redo, history management, and logging.

1. Install and Configure Git
Install Git
MacOS (using Homebrew)
brew install git
Windows
Download and install Git for Windows.
Accept the default options during installation.

Verify Git:

git --version
Configure Git Globals
Set your name and email so Git tracks your commits properly:

git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
Confirm the settings:

git config --list
Generate SSH Keys and Connect to GitHub
Only do this once per machine.

Generate a new SSH key:
ssh-keygen -t ed25519 -C "your_email@example.com"
(Press Enter at all prompts.)

Start the SSH agent:
eval "$(ssh-agent -s)"
Add the SSH private key to the agent:
ssh-add ~/.ssh/id_ed25519
Copy your SSH public key:
Mac/Linux:
cat ~/.ssh/id_ed25519.pub | pbcopy
Windows (Git Bash):
cat ~/.ssh/id_ed25519.pub | clip
Add the key to your GitHub account:

Go to GitHub SSH Settings
Click New SSH Key, paste the key, save.
Test the connection:

ssh -T git@github.com
You should see a success message.

2. Clone the Repository
Now you can safely clone the course project:

git clone (https://github.com/ds2464-lang/Midterm_Project.git)
cd Midterm_Project

3. Install Python 3.10+
Install Python
MacOS (Homebrew)
brew install python
Windows
Download and install Python for Windows.
Make sure you check the box Add Python to PATH during setup.

Verify Python:

python3 --version
or

python --version
Create and Activate a Virtual Environment
(Optional but recommended)

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate.bat  # Windows
Install Required Packages
pip install -r requirements.txt

4. Running the Project
python main.py or python3 main.py

5. Command-Line Interface
After running python main.py the initial greeting will show.
It will prompt you with the following:
  1. Enter command:
  2. First number:
  3. Second number:
You should receive the proper result.

Enter 'help' for more information
Enter 'history' to view history
Enter 'clear' to clear all history
Enter 'undo' to undo most recent operation
Enter 'redo' to redo most recently undone operation
Enter 'exit' to leave application

After entering 'exit' the farewell message will show.

6. Run tests

pytest or pytest -cov

7. GitHub Actions

Purpose of workflow tests is to:
Automate Testing, Code Quality Checks, Dependency Management, Automated Deployment, Documentation
and Build Checks, and Integration with GitHub Events.  In this instance it's to automatically
test and ensure an overal coverage percentage of 90% or above.

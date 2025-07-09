#!/bin/bash
# Script to set up a compatible Python environment for the backend

echo "🔧 Setting up compatible Python environment..."

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv is not installed. Please install it first:"
    echo "   brew install pyenv"
    echo "   Then add to your shell:"
    echo "   echo 'eval \"\$(pyenv init -)\"' >> ~/.zshrc"
    exit 1
fi

# Install Python 3.11 if not already installed
if ! pyenv versions | grep -q "3.11"; then
    echo "📦 Installing Python 3.11.8..."
    pyenv install 3.11.8
fi

# Create virtual environment with Python 3.11
echo "🐍 Creating virtual environment with Python 3.11..."
pyenv local 3.11.8
python -m venv venv_py311

# Activate the virtual environment
echo "✅ Activating virtual environment..."
source venv_py311/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements_py311.txt

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv_py311/bin/activate"
echo ""
echo "Then run the server with:"
echo "   python run.py"

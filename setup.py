#!/usr/bin/env python3
"""
Setup script for the Pharmacy RAG System.
Helps users get started quickly with proper configuration.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Print a welcome header."""
    print("=" * 60)
    print("PHARMACY RAG SYSTEM SETUP")
    print("=" * 60)
    print("Setting up your secure, local medical knowledge system...")
    print()


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Ollama is installed and accessible")
            return True
        else:
            print("Ollama is installed but may not be running")
            return False
    except FileNotFoundError:
        print("Ollama is not installed")
        print("Please install Ollama from: https://ollama.com")
        return False


def install_requirements():
    """Install Python requirements."""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def setup_data_directory():
    """Ensure data directory exists and show structure."""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("Created data directory")
    
    # Show existing subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if subdirs:
        print("Existing medical specialties:")
        for subdir in sorted(subdirs):
            file_count = len(list(subdir.glob("*.*")))
            print(f"   - {subdir.name}: {file_count} files")
    else:
        print("Data directory is empty - add your medical documents here")
    
    return True


def check_ollama_models():
    """Check if required models are available."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout
            if 'llama3.2:3b' in models:
                print("Recommended model (llama3.2:3b) is available")
                return True
            else:
                print("Recommended model not found")
                print("Run: ollama pull llama3.2:3b")
                return False
    except:
        pass
    
    return False


def run_system_check():
    """Run a quick system check."""
    print("Running system diagnostics...")
    
    # Check if vector database exists
    if os.path.exists("chroma"):
        print("Vector database found")
    else:
        print("Vector database not found - run 'python process_documents.py' to build it")
    
    # Check data directory
    data_files = list(Path("data").rglob("*.*"))
    print(f"Found {len(data_files)} files in data directory")
    
    return True


def show_next_steps():
    """Show next steps to the user."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("1. Add your medical documents to the 'data' directory")
    print("   - Supported formats: PDF, DOCX, TXT")
    print("   - Organize by medical specialty (optional)")
    print()
    
    print("2. Build the knowledge base:")
    print("   python process_documents.py")
    print()
    
    print("3. Test the system:")
    print("   python query_data.py \"What are the contraindications for beta-blockers?\"")
    print()
    
    print("4. Run comprehensive tests:")
    print("   python test_rag_system.py")
    print()
    
    print("🔒 Security Note: All processing happens locally - no data leaves your system!")
    print()


def main():
    """Main setup function."""
    print_header()
    
    # System checks
    checks_passed = 0
    total_checks = 5
    
    if check_python_version():
        checks_passed += 1
    
    if check_ollama():
        checks_passed += 1
    
    if install_requirements():
        checks_passed += 1
    
    if setup_data_directory():
        checks_passed += 1
    
    if check_ollama_models():
        checks_passed += 1
    
    print(f"\n System Check: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 3:
        print("System is ready for use!")
        run_system_check()
        show_next_steps()
    else:
        print("Please resolve the issues above before proceeding")
        print("\nFor help, see: https://github.com/Lapintam/pharmacy-rag-example")


if __name__ == "__main__":
    main() 
#!/bin/bash

# Cleanup script for multi-agent-system project
# Removes Python cache, temporary files, and build artifacts

echo "🧹 Cleaning up project..."

# Remove Python cache directories
echo "  → Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove Python bytecode files
echo "  → Removing .pyc, .pyo, .pyd files..."
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null

# Remove Python egg info
echo "  → Removing .egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove pytest cache
echo "  → Removing pytest cache..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Remove mypy cache
echo "  → Removing mypy cache..."
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

# Remove build directories
echo "  → Removing build directories..."
rm -rf build/ dist/ 2>/dev/null

# Remove .DS_Store files (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  → Removing .DS_Store files..."
    find . -name ".DS_Store" -delete 2>/dev/null
fi

echo "✓ Cleanup complete!"

#!/bin/bash
# Script to create GitHub repository and push code

echo "🌿 Weed Detection System - GitHub Setup"
echo "========================================"
echo ""

# Check if repo name is provided
REPO_NAME="${1:-weed-detection-yolo-fyp}"

echo "Repository will be created as: $REPO_NAME"
echo ""
echo "📋 Instructions:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository named: $REPO_NAME"
echo "3. DO NOT initialize with README, .gitignore, or license"
echo "4. Copy the repository URL"
echo ""
read -p "Press Enter after you've created the repository on GitHub..."

echo ""
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/$REPO_NAME.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ No repository URL provided. Exiting."
    exit 1
fi

echo ""
echo "🔗 Adding remote repository..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ Done! Your repository is now on GitHub:"
echo "   $REPO_URL"
echo ""
echo "📁 Files uploaded:"
echo "   - best.pt (trained model)"
echo "   - Weed_Detection_YOLO_FYP.ipynb (notebook)"
echo "   - webcam_inference.py"
echo "   - batch_inference.py"
echo "   - image_inference_gui_fixed.py"
echo "   - dataset/ (sample images)"
echo "   - batch_results/sample_annotated/ (sample results)"
echo "   - Documentation (README, guides)"
echo "   - model_analysis_results.json"


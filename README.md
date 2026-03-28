# BERTHCI - BiLSTM HCI Prompt Classification System

An advanced AI-powered chatbot with active learning capabilities, featuring a beautiful modern UI.

## Features

- ✨ **BiLSTM Neural Network** - Deep learning for prompt classification
- 🎯 **7 Intent Categories** - Translation, Calculator, Location, Education, Knowledge, Chat, Conversational
- 🔄 **Active Learning** - Learns from user corrections
- 💾 **SQLite Database** - Persistent storage for history and feedback
- 🎨 **Modern UI** - Beautiful dark theme with Three.js visualization
- 📊 **Real-time Stats** - Training metrics and performance tracking

## Installation

1. **Clone/Download the project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate datasets (if not present)**
```bash
python generate_datasets.py
```

4. **Train the model**
```bash
python train_model.py
```

5. **Run the Flask app**
```bash
python app.py
```

6. **Access the application**
Open browser at: `http://localhost:5000`

## Project Structure

```
BERTHCI_Project/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── templates/
│   └── index.html        # UI template
├── models/               # Trained models
├── CSV/                  # Datasets
└── berthci.db           # SQLite database
```

## Usage

### Chat with AI
Type natural language prompts:
- "Translate hello to Spanish"
- "Calculate 50 * 12"
- "Where is Paris"
- "Explain photosynthesis"

### Active Learning
1. If AI makes a mistake, click "Wrong?" button
2. Select correct intent
3. Click "RETRAIN" to update model

## API Endpoints

- `POST /predict` - Classify intent and get response
- `POST /feedback` - Submit correction
- `POST /train` - Retrain model with feedback
- `GET /history` - Get chat history
- `GET /stats` - Get training statistics

## Model Details

- **Architecture**: Bidirectional LSTM
- **Parameters**: ~130K
- **Accuracy**: ~94%
- **Categories**: 7 HCI intents

## Technologies

- **Backend**: Flask, TensorFlow, scikit-learn
- **Frontend**: Tailwind CSS, Three.js, GSAP
- **Database**: SQLite
- **ML**: BiLSTM, NLP

## License

MIT License - Educational purposes

## Author

Rashida Rezina

# MAIN SETUP FUNCTION
```def setup_project():
    """Create complete project structure"""
    
    print("\n" + "="*70)
    print(" BERTHCI PROJECT SETUP")
    print("="*70)
    
    base_dir = "BERTHCI_Project"
    
    # Create base directory
    if os.path.exists(base_dir):
        response = input(f"\\n{base_dir} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
        shutil.rmtree(base_dir)
    
    os.makedirs(base_dir)
    
    # Create subdirectories
    subdirs = [
        'templates',
        'static/css',
        'static/js',
        'models',
        'CSV',
        'Output/BERTHCI_Outputs/01_Architecture',
        'Output/BERTHCI_Outputs/02_Training',
        'Output/BERTHCI_Outputs/03_Performance',
        'Output/BERTHCI_Outputs/04_Data',
        'Output/BERTHCI_Outputs/05_Error',
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    # Create train_model.py
    with open(os.path.join(base_dir, 'train_model.py'), 'w') as f:
        f.write(TRAIN_MODEL_CONTENT)
    
    # Create requirements.txt
    with open(os.path.join(base_dir, 'requirements.txt'), 'w') as f:
        f.write(REQUIREMENTS_CONTENT)
    
    # Create README.md
    with open(os.path.join(base_dir, 'README.md'), 'w') as f:
        f.write(README_CONTENT)
    
    print(f"\\n✓ Project structure created: {base_dir}/")
    print("\\nNext steps:")
    print("1. Copy app.py to the project folder")
    print("2. Copy index.html to templates/")
    print("3. Generate datasets or copy existing CSV files to CSV/")
    print("4. Run: cd BERTHCI_Project && python train_model.py")
    print("5. Run: python app.py")
    print("6. Open: http://localhost:5000")
    
    print("\\n" + "="*70)
    print(PROJECT_STRUCTURE)
    print("="*70)

if __name__ == "__main__":
    setup_project()
# Automated Blog Content Tagger

An AI-powered system that automatically assigns relevant tags and keywords to blog content using Natural Language Processing (NLP) and machine learning techniques.

## 🚀 Features

- **AI-Powered Content Analysis**: Advanced NLP algorithms analyze blog content to understand context and meaning
- **Automatic Tag Generation**: Intelligently suggests relevant tags based on content analysis
- **Real-time Processing**: Get instant tag suggestions and content analysis in seconds
- **Comprehensive Statistics**: Word count, character count, reading time, and keyword extraction
- **Interactive Web Interface**: Beautiful, responsive UI for easy content analysis
- **Multi-label Classification**: Supports multiple tags per content piece
- **Keyword Extraction**: Identifies important keywords and their frequency

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, NLTK
- **NLP Libraries**: NLTK, TextBlob, TF-IDF Vectorization
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Data Processing**: pandas, numpy

## 📋 Prerequisites

Before running this project, make sure you have:

- Python 3.7 or higher
- pip (Python package installer)

## 🔧 Installation & Setup

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd automated-blog-content-tagger

# Or simply download the project files to your local directory
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- Flask 2.3.3
- Flask-CORS 4.0.0
- NLTK 3.8.1
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3
- beautifulsoup4 4.12.2
- textblob 0.17.1
- And other dependencies

## 🚀 Running the Application

### Step 1: Start the Application

```bash
python app.py
```

The application will:
1. Download required NLTK data automatically
2. Train the machine learning model with sample data
3. Start the Flask server on `http://localhost:5000`

### Step 2: Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## 📖 How to Use

### 1. **Train the Model** (Optional)
- Click the "Train Model" button to retrain the AI model
- This process takes a few seconds and uses sample blog data

### 2. **Analyze Your Content**
- Enter your blog title (optional)
- Paste your blog content in the text area
- Click "Analyze & Tag Content"

### 3. **View Results**
The system will provide:
- **Content Statistics**: Word count, character count, sentences, reading time
- **Predicted Tags**: AI-suggested tags with confidence scores
- **Top Keywords**: Most frequent and relevant keywords

## 🎯 Supported Tags

The system can identify and suggest tags from these categories:

- **Technology**: programming, web-development, data-science
- **AI/ML**: artificial-intelligence, machine-learning
- **Business**: startup, finance, marketing
- **Lifestyle**: health, travel, food, productivity
- **Creative**: design, photography, music
- **Education**: career, education
- **Entertainment**: sports, entertainment
- **News**: news, politics

## 🔧 API Endpoints

The application provides REST API endpoints:

### POST `/train`
Retrain the machine learning model
```bash
curl -X POST http://localhost:5000/train
```

### POST `/predict`
Predict tags for given content
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"content": "Your blog content here"}'
```

### POST `/analyze`
Comprehensive content analysis
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"title": "Blog Title", "content": "Your blog content here"}'
```

## 🧠 How It Works

### 1. **Text Preprocessing**
- Converts text to lowercase
- Removes special characters and digits
- Tokenizes and lemmatizes words
- Removes stopwords

### 2. **Feature Extraction**
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Extracts n-grams (1-gram and 2-gram features)
- Filters rare and common terms

### 3. **Machine Learning Model**
- Multi-output Random Forest classifier
- Trained on sample blog data with multiple tags
- Predicts probability scores for each tag category

### 4. **Tag Selection**
- Ranks tags by confidence scores
- Filters tags with low confidence (< 10%)
- Returns top 5-8 most relevant tags

## 📊 Sample Data

The system comes with pre-loaded sample blog data covering:
- Programming tutorials
- Technology articles
- Lifestyle content
- Business insights
- Creative topics

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Interface**: Clean, professional design with gradients and animations
- **Real-time Feedback**: Loading indicators and progress updates
- **Interactive Elements**: Hover effects and smooth transitions
- **Alert System**: Success and error notifications

## 🔍 Troubleshooting

### Common Issues:

1. **NLTK Data Download Error**
   ```
   Solution: The app automatically downloads NLTK data on first run
   If it fails, manually run:
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

2. **Port Already in Use**
   ```
   Solution: Change the port in app.py:
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

3. **Dependencies Installation Issues**
   ```
   Solution: Upgrade pip and try again:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider:
- Using a WSGI server like Gunicorn
- Setting up environment variables
- Using a reverse proxy like Nginx
- Implementing proper logging and monitoring

## 📈 Performance

- **Training Time**: ~5-10 seconds for sample data
- **Prediction Time**: <1 second for typical blog content
- **Memory Usage**: ~100-200MB for the complete application
- **Supported Content Length**: Up to 10,000 words

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new tag categories
- Improving the machine learning model
- Enhancing the UI/UX
- Adding new features
- Fixing bugs

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Educational Value

This project demonstrates:
- **Natural Language Processing** concepts and implementation
- **Machine Learning** for text classification
- **Multi-label classification** techniques
- **Web application development** with Flask
- **Frontend development** with modern web technologies
- **API design** and RESTful services

Perfect for learning NLP, machine learning, and full-stack web development!

---

**Happy Blogging! 📝✨**


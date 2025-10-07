from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data download completed!")

class BlogContentTagger:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # No predefined categories - pure content-based analysis
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_keywords_from_text(self, text):
        """Extract meaningful keywords from text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq
    
    def extract_meaningful_phrases(self, text):
        """Extract meaningful phrases and n-grams from text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Extract meaningful bigrams and trigrams
        phrases = []
        
        # Common words that don't make good tags
        skip_words = {'this', 'that', 'with', 'from', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'how', 'why', 'will', 'would', 'could', 'should', 'have', 'has', 'had', 'been', 'being', 'get', 'got', 'just', 'only', 'also', 'more', 'most', 'some', 'many', 'much', 'very', 'really', 'quite', 'rather'}
        
        # Bigrams
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i+1]
            if (len(word1) > 2 and len(word2) > 2 and 
                word1 not in skip_words and word2 not in skip_words and
                not word1.isdigit() and not word2.isdigit()):
                phrases.append(f"{word1} {word2}")
        
        # Trigrams (more selective)
        for i in range(len(words) - 2):
            word1, word2, word3 = words[i], words[i+1], words[i+2]
            if (all(len(w) > 2 for w in [word1, word2, word3]) and
                all(w not in skip_words for w in [word1, word2, word3]) and
                not any(w.isdigit() for w in [word1, word2, word3])):
                phrases.append(f"{word1} {word2} {word3}")
        
        return phrases
    
    def analyze_content_themes(self, text):
        """Analyze content to identify main themes using NLP techniques"""
        text_lower = text.lower()
        
        # Extract key entities and concepts
        themes = {}
        
        # Travel-related analysis
        travel_indicators = ['city', 'pass', 'tour', 'museum', 'visit', 'ticket', 'admission', 'crossing', 
                           'guided', 'attraction', 'sightseeing', 'destination', 'travel', 'tourism', 
                           'ferris', 'wheel', 'stadium', 'cruise', 'hotel', 'flight', 'vacation']
        
        travel_score = sum(1 for indicator in travel_indicators if indicator in text_lower)
        if travel_score > 0:
            themes['travel'] = min(travel_score / 10.0, 1.0)
        
        # Technology-related analysis
        tech_indicators = ['programming', 'software', 'code', 'developer', 'app', 'website', 'computer', 
                          'tech', 'digital', 'algorithm', 'python', 'javascript', 'html', 'css', 'react', 
                          'node', 'database', 'api', 'framework', 'coding', 'development', 'sql', 'server',
                          'azure', 'cloud', 'microsoft', 'oracle', 'mysql', 'postgresql', 'nosql', 'data',
                          'query', 'processing', 'performance', 'workload', 'compatibility',
                          'transact', 'managed', 'instance', 'workloads', 'features', 'broadcast']
        
        tech_score = sum(1 for indicator in tech_indicators if indicator in text_lower)
        if tech_score > 0:
            themes['technology'] = min(tech_score / 10.0, 1.0)
        
        # Business-related analysis
        business_indicators = ['business', 'company', 'startup', 'entrepreneur', 'revenue', 'profit', 
                              'market', 'industry', 'strategy', 'management', 'leadership', 'corporate', 
                              'finance', 'investment', 'funding', 'capital', 'economy']
        
        business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
        if business_score > 0:
            themes['business'] = min(business_score / 10.0, 1.0)
        
        # Health and lifestyle analysis
        health_indicators = ['health', 'medical', 'doctor', 'hospital', 'treatment', 'medicine', 
                           'wellness', 'fitness', 'exercise', 'nutrition', 'mental', 'physical', 
                           'lifestyle', 'daily', 'routine', 'habits', 'personal', 'balance', 'mindfulness',
                           'weekend', 'party', 'date', 'shoes', 'sale', 'fashion', 'style', 'beauty',
                           'home', 'family', 'friends', 'hang', 'out', 'fun', 'relax', 'enjoy']
        
        health_score = sum(1 for indicator in health_indicators if indicator in text_lower)
        if health_score > 0:
            themes['health-lifestyle'] = min(health_score / 10.0, 1.0)
        
        # Education and learning
        education_indicators = ['education', 'learning', 'school', 'university', 'student', 'teacher', 
                               'course', 'study', 'knowledge', 'academic', 'research', 'training', 
                               'tutorial', 'guide', 'lesson', 'instruction']
        
        education_score = sum(1 for indicator in education_indicators if indicator in text_lower)
        if education_score > 0:
            themes['education'] = min(education_score / 10.0, 1.0)
        
        # Creative and arts
        creative_indicators = ['design', 'creative', 'art', 'graphic', 'visual', 'aesthetic', 'brand', 
                              'logo', 'typography', 'color', 'layout', 'beauty', 'photography', 'camera', 
                              'image', 'picture', 'music', 'film', 'movie', 'entertainment']
        
        creative_score = sum(1 for indicator in creative_indicators if indicator in text_lower)
        if creative_score > 0:
            themes['creative-arts'] = min(creative_score / 10.0, 1.0)
        
        return themes
    
    def extract_domain_specific_tags(self, text):
        """Extract domain-specific technology tags"""
        text_lower = text.lower()
        domain_tags = []
        
        # Database and SQL technologies
        if any(term in text_lower for term in ['azure sql', 'sql server', 'sql database']):
            if 'azure sql' in text_lower:
                domain_tags.append(('azure-sql', 0.9))
            if 'sql server' in text_lower:
                domain_tags.append(('sql-server', 0.8))
            if 'sql database' in text_lower:
                domain_tags.append(('sql-database', 0.7))
        
        # Cloud platforms
        if 'azure' in text_lower:
            domain_tags.append(('azure', 0.8))
        if 'aws' in text_lower or 'amazon web services' in text_lower:
            domain_tags.append(('aws', 0.8))
        if 'google cloud' in text_lower or 'gcp' in text_lower:
            domain_tags.append(('google-cloud', 0.8))
        
        # Programming languages
        if 'python' in text_lower:
            domain_tags.append(('python', 0.7))
        if 'javascript' in text_lower or 'js' in text_lower:
            domain_tags.append(('javascript', 0.7))
        if 'java' in text_lower:
            domain_tags.append(('java', 0.7))
        if 'c#' in text_lower or 'csharp' in text_lower:
            domain_tags.append(('csharp', 0.7))
        
        # Web technologies
        if 'react' in text_lower:
            domain_tags.append(('react', 0.6))
        if 'node.js' in text_lower or 'nodejs' in text_lower:
            domain_tags.append(('nodejs', 0.6))
        if 'angular' in text_lower:
            domain_tags.append(('angular', 0.6))
        if 'vue' in text_lower:
            domain_tags.append(('vue', 0.6))
        
        # Database technologies
        if 'mysql' in text_lower:
            domain_tags.append(('mysql', 0.7))
        if 'postgresql' in text_lower or 'postgres' in text_lower:
            domain_tags.append(('postgresql', 0.7))
        if 'mongodb' in text_lower or 'mongo' in text_lower:
            domain_tags.append(('mongodb', 0.7))
        if 'redis' in text_lower:
            domain_tags.append(('redis', 0.6))
        
        # AI/ML technologies (only if clearly AI/ML content with strong indicators)
        ai_strong_indicators = ['machine learning', 'neural network', 'deep learning', 'tensorflow', 'pytorch', 'artificial intelligence', 'computer vision', 'natural language processing', 'nlp', 'chatbot', 'algorithm training', 'data model', 'predictive analytics']
        
        # Check for lifestyle content that shouldn't be tagged as AI
        lifestyle_indicators = ['weekend', 'party', 'date', 'shoes', 'sale', 'fashion', 'style', 'beauty', 'home', 'family', 'friends', 'hang', 'out', 'fun', 'relax', 'enjoy', 'personal', 'lifestyle', 'daily']
        is_lifestyle_content = any(indicator in text_lower for indicator in lifestyle_indicators)
        
        if any(indicator in text_lower for indicator in ai_strong_indicators) and not is_lifestyle_content:
            if 'machine learning' in text_lower or 'ml' in text_lower:
                domain_tags.append(('machine-learning', 0.8))
            if 'artificial intelligence' in text_lower or 'ai' in text_lower:
                domain_tags.append(('artificial-intelligence', 0.8))
            if 'tensorflow' in text_lower:
                domain_tags.append(('tensorflow', 0.6))
            if 'pytorch' in text_lower:
                domain_tags.append(('pytorch', 0.6))
        
        # DevOps and tools
        if 'docker' in text_lower:
            domain_tags.append(('docker', 0.6))
        if 'kubernetes' in text_lower or 'k8s' in text_lower:
            domain_tags.append(('kubernetes', 0.6))
        if 'jenkins' in text_lower:
            domain_tags.append(('jenkins', 0.5))
        if 'git' in text_lower:
            domain_tags.append(('git', 0.5))
        
        return domain_tags
    
    def remove_redundant_tags(self, tag_candidates):
        """Remove redundant and overlapping tags with aggressive filtering"""
        if not tag_candidates:
            return []
        
        # Sort by confidence (highest first)
        tag_candidates.sort(key=lambda x: x[1], reverse=True)
        
        filtered_tags = []
        seen_words = set()
        seen_concepts = set()
        
        # Define concept groups to avoid redundancy
        concept_groups = {
            'sql': ['sql', 'database', 'azure-sql', 'sql-server', 'sql-database'],
            'azure': ['azure', 'azure-sql'],
            'query': ['query', 'intelligent-query', 'query-processing'],
            'cloud': ['azure', 'aws', 'google-cloud', 'cloud'],
            'ai': ['artificial-intelligence', 'ai', 'machine-learning', 'ml'],
            'web': ['react', 'angular', 'vue', 'nodejs', 'javascript'],
            'db': ['mysql', 'postgresql', 'mongodb', 'redis', 'database']
        }
        
        for tag, confidence in tag_candidates:
            tag_lower = tag.lower()
            tag_words = set(tag_lower.split())
            
            # Check if tag belongs to a concept group
            tag_concept = None
            for concept, terms in concept_groups.items():
                if any(term in tag_lower for term in terms):
                    tag_concept = concept
                    break
            
            # Skip if we already have a tag from the same concept group
            if tag_concept and tag_concept in seen_concepts:
                continue
            
            # Skip if this tag is too similar to an already selected tag
            is_redundant = False
            for existing_tag, _ in filtered_tags:
                existing_words = set(existing_tag.lower().split())
                
                # More aggressive overlap detection (50% instead of 70%)
                if len(tag_words & existing_words) >= len(tag_words) * 0.5:
                    is_redundant = True
                    break
                
                # Check if one tag is contained in another
                if tag_words.issubset(existing_words) or existing_words.issubset(tag_words):
                    is_redundant = True
                    break
                
                # Check for shared root words
                if any(word in existing_words for word in tag_words if len(word) > 3):
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered_tags.append((tag, confidence))
                
                # Track words and concepts
                seen_words.update(tag_words)
                if tag_concept:
                    seen_concepts.add(tag_concept)
        
        return filtered_tags
    
    def generate_dynamic_tags(self, text, top_k=6):
        """Generate dynamic tags based on pure content analysis using NLP"""
        # Analyze content themes
        themes = self.analyze_content_themes(text)
        
        # Extract meaningful phrases
        phrases = self.extract_meaningful_phrases(text)
        
        # Create tag candidates from themes and phrases
        tag_candidates = []
        
        # Add theme-based tags (these are always good)
        for theme, score in themes.items():
            if score > 0.15:  # Higher threshold for themes
                tag_candidates.append((theme, score))
        
        # Add domain-specific technology tags
        domain_tags = self.extract_domain_specific_tags(text)
        tag_candidates.extend(domain_tags)
        
        # Add phrase-based tags (top frequent meaningful phrases)
        phrase_freq = {}
        for phrase in phrases:
            phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
        
        # Get top phrases that appear multiple times and are meaningful
        frequent_phrases = [(phrase, count) for phrase, count in phrase_freq.items() 
                           if count > 1 and len(phrase.split()) >= 2 and len(phrase.split()) <= 3]
        frequent_phrases.sort(key=lambda x: x[1], reverse=True)
        
        # Add top phrases as tags (with better filtering)
        added_phrases = 0
        for phrase, count in frequent_phrases:
            if added_phrases >= 3:  # Limit to 3 phrase tags
                break
                
            # Skip phrases that look like personal names
            words = phrase.split()
            if (len(words) == 2 and 
                any(word[0].isupper() for word in words) and
                not any(word in ['city', 'pass', 'tour', 'museum', 'guide', 'travel', 'hotel', 'restaurant', 'park', 'beach', 'mountain', 'river', 'lake', 'island', 'country', 'state', 'region'] for word in words)):
                continue
            
            # Skip phrases with too many proper nouns
            if sum(1 for word in words if word[0].isupper()) > len(words) * 0.5:
                continue
                
            confidence = min(count / 5.0, 0.8)  # Cap confidence for phrases
            tag_candidates.append((phrase, confidence))
            added_phrases += 1
        
        # Remove redundant tags
        filtered_tags = self.remove_redundant_tags(tag_candidates)
        
        # Return top results (limit to 4 for better quality)
        return filtered_tags[:min(top_k, 4)]
    
    def train_model(self):
        """Initialize the content tagger with NLP-based dynamic analysis"""
        print("NLP-based content tagger initialized!")
        print("Using advanced text analysis for dynamic tag generation.")
        return True
    
    def predict_tags(self, text, top_k=5):
        """Predict tags using content-based analysis"""
        return self.generate_dynamic_tags(text, top_k)

# Initialize the tagger
tagger = BlogContentTagger()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Automated Blog Content Tagger</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .main-container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                margin: 20px auto;
                padding: 30px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .header h1 {
                color: #2c3e50;
                font-weight: 700;
                margin-bottom: 10px;
            }
            
            .form-section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 30px;
            }
            
            .form-control {
                border-radius: 10px;
                border: 2px solid #e9ecef;
                padding: 12px 15px;
                transition: all 0.3s ease;
            }
            
            .form-control:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }
            
            .btn-primary {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border: none;
                border-radius: 10px;
                padding: 12px 30px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .results-section {
                background: #ffffff;
                border-radius: 15px;
                padding: 25px;
                margin-top: 30px;
                border: 1px solid #e9ecef;
                display: none;
            }
            
            .tag-badge {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                margin: 5px;
                display: inline-block;
                font-weight: 500;
                font-size: 0.9em;
            }
            
            .stats-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                text-align: center;
            }
            
            .stats-number {
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 5px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner-border {
                color: #667eea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="main-container">
                <div class="header">
                    <h1><i class="fas fa-robot text-primary"></i> Automated Content Tagging</h1>
                    <p>AI-powered system for automatically generating relevant tags for your blog content</p>
                </div>

                <div class="form-section">
                    <h3><i class="fas fa-edit text-primary"></i> Content Analysis</h3>
                    <form id="contentForm">
                        <div class="mb-3">
                            <label for="title" class="form-label">Blog Title (Optional)</label>
                            <input type="text" class="form-control" id="title" placeholder="Enter your blog title...">
                        </div>
                        <div class="mb-3">
                            <label for="content" class="form-label">Blog Content *</label>
                            <textarea class="form-control" id="content" rows="8" placeholder="Paste your blog content here for automatic tagging..." required></textarea>
                        </div>
                        <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                            <button type="submit" class="btn btn-primary">
                                <i class="fas fa-magic"></i> Analyze & Tag Content
                            </button>
                        </div>
                    </form>
                </div>

                <div class="loading" id="loading">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Analyzing your content...</p>
                </div>

                <div class="results-section" id="results">
                    <h3><i class="fas fa-chart-line text-primary"></i> Analysis Results</h3>
                    
                    <div class="row mb-4" id="statsContainer">
                    </div>
                    
                    <div class="mb-4">
                        <h4><i class="fas fa-tags text-success"></i> Predicted Tags</h4>
                        <div id="tagsContainer">
                        </div>
                    </div>
                    
                    <div class="mb-4">
                        <h4><i class="fas fa-key text-info"></i> Top Keywords</h4>
                        <div id="keywordsContainer">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const form = document.getElementById('contentForm');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');

                form.addEventListener('submit', function(e) {
                    e.preventDefault();
                    analyzeContent();
                });

                function analyzeContent() {
                    const title = document.getElementById('title').value;
                    const content = document.getElementById('content').value;

                    if (!content.trim()) {
                        alert('Please enter some content to analyze.');
                        return;
                    }

                    loading.style.display = 'block';
                    results.style.display = 'none';

                    fetch('/api/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            title: title,
                            content: content
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        loading.style.display = 'none';
                        
                        if (data.success) {
                            displayResults(data.analysis);
                            results.style.display = 'block';
                            results.scrollIntoView({ behavior: 'smooth' });
                        } else {
                            alert('Error analyzing content: ' + data.message);
                        }
                    })
                    .catch(error => {
                        loading.style.display = 'none';
                        alert('Error: ' + error.message);
                    });
                }

                function displayResults(analysis) {
                    const statsContainer = document.getElementById('statsContainer');
                    statsContainer.innerHTML = `
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number">${analysis.word_count}</div>
                                <div class="stats-label">Words</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number">${analysis.char_count}</div>
                                <div class="stats-label">Characters</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number">${analysis.sentence_count}</div>
                                <div class="stats-label">Sentences</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <div class="stats-number">${analysis.reading_time}</div>
                                <div class="stats-label">Min Read</div>
                            </div>
                        </div>
                    `;

                    const tagsContainer = document.getElementById('tagsContainer');
                    tagsContainer.innerHTML = '';
                    
                    if (analysis.predicted_tags && analysis.predicted_tags.length > 0) {
                        analysis.predicted_tags.forEach(tag => {
                            const badge = document.createElement('span');
                            badge.className = 'tag-badge';
                            badge.textContent = `${tag[0]} (${(tag[1] * 100).toFixed(1)}%)`;
                            tagsContainer.appendChild(badge);
                        });
                    } else {
                        tagsContainer.innerHTML = '<p class="text-muted">No tags predicted with sufficient confidence.</p>';
                    }

                    const keywordsContainer = document.getElementById('keywordsContainer');
                    keywordsContainer.innerHTML = '';
                    
                    if (analysis.top_keywords && analysis.top_keywords.length > 0) {
                        analysis.top_keywords.forEach(keyword => {
                            const keywordDiv = document.createElement('div');
                            keywordDiv.className = 'keyword-item';
                            keywordDiv.textContent = `${keyword[0]} (${keyword[1]})`;
                            keywordsContainer.appendChild(keywordDiv);
                        });
                    } else {
                        keywordsContainer.innerHTML = '<p class="text-muted">No keywords extracted.</p>';
                    }
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the model"""
    try:
        result = tagger.train_model()
        return jsonify({
            'success': True,
            'message': 'Model trained successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def predict_tags():
    """Predict tags for given content"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Content is required'
            })
        
        # Predict tags
        predictions = tagger.predict_tags(content)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'content_length': len(content)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """Comprehensive content analysis"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        title = data.get('title', '')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Content is required'
            })
        
        # Combine title and content for analysis
        full_text = f"{title} {content}"
        
        # Predict tags using dynamic content analysis
        predictions = tagger.generate_dynamic_tags(full_text, top_k=8)
        
        # Basic text statistics
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = len(content.split('.'))
        
        # Keyword extraction (meaningful content-based)
        word_freq = tagger.extract_keywords_from_text(content)
        
        # Filter out common words and get top meaningful keywords
        meaningful_keywords = []
        for word, freq in word_freq.items():
            if len(word) > 3 and freq > 1:  # Must appear more than once
                meaningful_keywords.append((word, freq))
        
        top_keywords = sorted(meaningful_keywords, key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'success': True,
            'analysis': {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'predicted_tags': predictions,
                'top_keywords': top_keywords,
                'reading_time': round(word_count / 200, 1)  # Assuming 200 WPM
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)

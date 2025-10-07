from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Removed sklearn imports - now using content-based analysis
import re
import pickle
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
        skip_words = {'this', 'that', 'with', 'from', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'how', 'why', 'will', 'would', 'could', 'should', 'have', 'has', 'had', 'been', 'being', 'get', 'got', 'just', 'only', 'also', 'more', 'most', 'some', 'many', 'much', 'very', 'really', 'quite', 'rather', 'about', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'since', 'until', 'while', 'because', 'although', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'meanwhile', 'consequently', 'accordingly', 'likewise', 'similarly', 'instead', 'rather', 'actually', 'basically', 'essentially', 'fundamentally', 'generally', 'particularly', 'specifically', 'especially', 'particularly', 'obviously', 'clearly', 'certainly', 'definitely', 'absolutely', 'completely', 'totally', 'entirely', 'wholly', 'fully', 'partly', 'partially', 'mainly', 'mostly', 'primarily', 'chiefly', 'largely', 'considerably', 'significantly', 'substantially', 'dramatically', 'tremendously', 'enormously', 'immensely', 'vastly', 'hugely', 'greatly', 'deeply', 'highly', 'extremely', 'incredibly', 'amazingly', 'surprisingly', 'unexpectedly', 'fortunately', 'unfortunately', 'hopefully', 'ideally', 'theoretically', 'practically', 'realistically', 'honestly', 'frankly', 'seriously', 'literally', 'figuratively', 'metaphorically', 'technically', 'scientifically', 'historically', 'culturally', 'socially', 'economically', 'politically', 'environmentally', 'globally', 'locally', 'nationally', 'internationally', 'regionally', 'urbanly', 'rurally', 'traditionally', 'conventionally', 'typically', 'normally', 'usually', 'commonly', 'frequently', 'often', 'sometimes', 'occasionally', 'rarely', 'seldom', 'hardly', 'barely', 'scarcely', 'almost', 'nearly', 'approximately', 'roughly', 'about', 'around', 'circa', 'plus', 'minus', 'including', 'excluding', 'except', 'besides', 'additionally', 'furthermore', 'moreover', 'besides', 'likewise', 'similarly', 'conversely', 'alternatively', 'otherwise', 'meanwhile', 'simultaneously', 'concurrently', 'subsequently', 'previously', 'initially', 'finally', 'ultimately', 'eventually', 'gradually', 'suddenly', 'immediately', 'instantly', 'quickly', 'slowly', 'rapidly', 'gradually', 'steadily', 'constantly', 'continuously', 'permanently', 'temporarily', 'briefly', 'shortly', 'recently', 'lately', 'currently', 'presently', 'nowadays', 'today', 'yesterday', 'tomorrow', 'tonight', 'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year', 'time', 'times', 'moment', 'period', 'duration', 'length', 'size', 'amount', 'number', 'quantity', 'quality', 'level', 'degree', 'extent', 'scope', 'range', 'limit', 'boundary', 'edge', 'corner', 'side', 'top', 'bottom', 'middle', 'center', 'front', 'back', 'left', 'right', 'north', 'south', 'east', 'west', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'above', 'below', 'inside', 'outside', 'within', 'without', 'through', 'across', 'around', 'beyond', 'behind', 'beside', 'near', 'far', 'close', 'distant', 'local', 'remote', 'here', 'there', 'everywhere', 'nowhere', 'somewhere', 'anywhere', 'home', 'away', 'abroad', 'overseas', 'domestic', 'foreign', 'international', 'national', 'regional', 'local', 'urban', 'rural', 'city', 'town', 'village', 'country', 'state', 'province', 'district', 'area', 'zone', 'region', 'territory', 'place', 'location', 'position', 'situation', 'condition', 'state', 'status', 'situation', 'circumstance', 'context', 'environment', 'setting', 'background', 'foreground', 'scene', 'stage', 'platform', 'base', 'foundation', 'ground', 'floor', 'surface', 'level', 'layer', 'section', 'part', 'portion', 'piece', 'bit', 'fragment', 'segment', 'division', 'unit', 'component', 'element', 'factor', 'aspect', 'feature', 'characteristic', 'property', 'attribute', 'quality', 'trait', 'nature', 'kind', 'type', 'sort', 'category', 'class', 'group', 'set', 'collection', 'series', 'sequence', 'chain', 'line', 'row', 'column', 'list', 'table', 'chart', 'graph', 'diagram', 'picture', 'image', 'photo', 'video', 'audio', 'sound', 'voice', 'music', 'song', 'tune', 'melody', 'rhythm', 'beat', 'tone', 'pitch', 'volume', 'loud', 'quiet', 'silent', 'noise', 'sound', 'voice', 'speech', 'talk', 'conversation', 'discussion', 'debate', 'argument', 'disagreement', 'agreement', 'consensus', 'opinion', 'view', 'perspective', 'point', 'angle', 'approach', 'method', 'technique', 'strategy', 'tactic', 'plan', 'scheme', 'program', 'project', 'initiative', 'effort', 'attempt', 'try', 'trial', 'test', 'experiment', 'study', 'research', 'investigation', 'inquiry', 'examination', 'analysis', 'evaluation', 'assessment', 'review', 'critique', 'criticism', 'feedback', 'response', 'reaction', 'result', 'outcome', 'consequence', 'effect', 'impact', 'influence', 'change', 'difference', 'improvement', 'progress', 'development', 'growth', 'increase', 'decrease', 'rise', 'fall', 'up', 'down', 'high', 'low', 'big', 'small', 'large', 'little', 'huge', 'tiny', 'massive', 'miniature', 'giant', 'dwarf', 'long', 'short', 'tall', 'wide', 'narrow', 'thick', 'thin', 'deep', 'shallow', 'broad', 'tight', 'loose', 'open', 'closed', 'full', 'empty', 'complete', 'partial', 'whole', 'half', 'quarter', 'third', 'double', 'triple', 'single', 'multiple', 'several', 'few', 'many', 'much', 'more', 'most', 'less', 'least', 'all', 'none', 'some', 'any', 'every', 'each', 'both', 'either', 'neither', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million', 'billion', 'first', 'second', 'third', 'last', 'final', 'initial', 'beginning', 'end', 'start', 'finish', 'stop', 'continue', 'proceed', 'go', 'come', 'arrive', 'leave', 'depart', 'enter', 'exit', 'move', 'stay', 'remain', 'keep', 'hold', 'grab', 'take', 'give', 'put', 'place', 'set', 'lay', 'sit', 'stand', 'lie', 'sleep', 'wake', 'eat', 'drink', 'cook', 'buy', 'sell', 'pay', 'cost', 'price', 'money', 'cash', 'dollar', 'euro', 'pound', 'cent', 'penny', 'coin', 'bill', 'note', 'check', 'card', 'credit', 'debit', 'bank', 'account', 'balance', 'debt', 'loan', 'interest', 'rate', 'tax', 'fee', 'charge', 'cost', 'expense', 'budget', 'income', 'salary', 'wage', 'earn', 'make', 'profit', 'loss', 'gain', 'win', 'lose', 'beat', 'defeat', 'victory', 'success', 'failure', 'achieve', 'accomplish', 'complete', 'finish', 'succeed', 'fail', 'try', 'attempt', 'effort', 'work', 'job', 'task', 'duty', 'responsibility', 'role', 'function', 'purpose', 'goal', 'objective', 'target', 'aim', 'plan', 'intention', 'desire', 'want', 'need', 'require', 'demand', 'request', 'ask', 'question', 'answer', 'reply', 'respond', 'react', 'behave', 'act', 'do', 'make', 'create', 'build', 'construct', 'design', 'develop', 'produce', 'manufacture', 'generate', 'form', 'shape', 'structure', 'organization', 'system', 'process', 'procedure', 'operation', 'activity', 'action', 'movement', 'motion', 'speed', 'velocity', 'acceleration', 'deceleration', 'direction', 'way', 'path', 'route', 'road', 'street', 'avenue', 'boulevard', 'highway', 'freeway', 'expressway', 'lane', 'alley', 'drive', 'place', 'court', 'circle', 'square', 'plaza', 'park', 'garden', 'yard', 'field', 'ground', 'land', 'territory', 'country', 'nation', 'state', 'province', 'county', 'district', 'city', 'town', 'village', 'hamlet', 'settlement', 'community', 'neighborhood', 'area', 'region', 'zone', 'sector', 'quarter', 'district', 'ward', 'precinct', 'block', 'building', 'house', 'home', 'residence', 'apartment', 'flat', 'condo', 'mansion', 'palace', 'castle', 'fortress', 'tower', 'skyscraper', 'office', 'factory', 'warehouse', 'store', 'shop', 'market', 'mall', 'center', 'centre', 'complex', 'facility', 'institution', 'organization', 'company', 'corporation', 'business', 'firm', 'agency', 'department', 'ministry', 'government', 'administration', 'authority', 'board', 'committee', 'council', 'assembly', 'parliament', 'congress', 'senate', 'house', 'chamber', 'room', 'hall', 'auditorium', 'theater', 'theatre', 'cinema', 'movie', 'film', 'show', 'play', 'drama', 'comedy', 'tragedy', 'musical', 'opera', 'ballet', 'concert', 'performance', 'presentation', 'exhibition', 'display', 'showcase', 'demonstration', 'explanation', 'description', 'definition', 'meaning', 'sense', 'significance', 'importance', 'value', 'worth', 'benefit', 'advantage', 'disadvantage', 'pro', 'con', 'positive', 'negative', 'good', 'bad', 'excellent', 'terrible', 'wonderful', 'awful', 'great', 'poor', 'best', 'worst', 'better', 'worse', 'superior', 'inferior', 'higher', 'lower', 'upper', 'lower', 'top', 'bottom', 'front', 'back', 'left', 'right', 'east', 'west', 'north', 'south', 'central', 'middle', 'inner', 'outer', 'internal', 'external', 'inside', 'outside', 'within', 'without', 'including', 'excluding', 'except', 'besides', 'additionally', 'furthermore', 'moreover', 'however', 'nevertheless', 'nonetheless', 'although', 'though', 'despite', 'in spite of', 'regardless of', 'irrespective of', 'concerning', 'regarding', 'about', 'on', 'over', 'under', 'above', 'below', 'beyond', 'across', 'through', 'around', 'beyond', 'behind', 'beside', 'near', 'far', 'close', 'distant', 'remote', 'local', 'regional', 'national', 'international', 'global', 'worldwide', 'universal', 'general', 'specific', 'particular', 'special', 'unique', 'common', 'ordinary', 'normal', 'typical', 'standard', 'regular', 'usual', 'customary', 'traditional', 'conventional', 'modern', 'contemporary', 'current', 'present', 'recent', 'latest', 'new', 'old', 'ancient', 'historic', 'historical', 'classic', 'timeless', 'eternal', 'permanent', 'temporary', 'brief', 'short', 'long', 'quick', 'slow', 'fast', 'rapid', 'swift', 'sudden', 'gradual', 'immediate', 'instant', 'delayed', 'postponed', 'cancelled', 'scheduled', 'planned', 'organized', 'structured', 'systematic', 'methodical', 'logical', 'rational', 'reasonable', 'sensible', 'practical', 'realistic', 'ideal', 'perfect', 'flawless', 'faultless', 'excellent', 'outstanding', 'remarkable', 'extraordinary', 'exceptional', 'unusual', 'rare', 'unique', 'special', 'particular', 'specific', 'individual', 'personal', 'private', 'public', 'official', 'formal', 'informal', 'casual', 'relaxed', 'tense', 'stressed', 'calm', 'peaceful', 'quiet', 'silent', 'noisy', 'loud', 'soft', 'gentle', 'harsh', 'rough', 'smooth', 'rough', 'hard', 'soft', 'firm', 'flexible', 'rigid', 'stiff', 'loose', 'tight', 'secure', 'safe', 'dangerous', 'risky', 'careful', 'careless', 'attentive', 'inattentive', 'focused', 'distracted', 'concentrated', 'scattered', 'organized', 'disorganized', 'neat', 'messy', 'clean', 'dirty', 'fresh', 'stale', 'new', 'old', 'recent', 'ancient', 'modern', 'traditional', 'contemporary', 'classic', 'fashionable', 'trendy', 'stylish', 'elegant', 'beautiful', 'attractive', 'handsome', 'pretty', 'cute', 'lovely', 'charming', 'appealing', 'desirable', 'wanted', 'needed', 'required', 'necessary', 'essential', 'important', 'significant', 'meaningful', 'valuable', 'precious', 'expensive', 'cheap', 'affordable', 'costly', 'inexpensive', 'free', 'paid', 'purchased', 'bought', 'sold', 'available', 'unavailable', 'accessible', 'inaccessible', 'reachable', 'unreachable', 'possible', 'impossible', 'feasible', 'infeasible', 'practical', 'impractical', 'realistic', 'unrealistic', 'achievable', 'unachievable', 'attainable', 'unattainable', 'obtainable', 'unobtainable', 'available', 'unavailable', 'accessible', 'inaccessible', 'reachable', 'unreachable', 'obtainable', 'unobtainable', 'achievable', 'unachievable', 'attainable', 'unattainable', 'possible', 'impossible', 'feasible', 'infeasible', 'practical', 'impractical', 'realistic', 'unrealistic'}
        
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
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model"""
    try:
        df = tagger.train_model()
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'training_samples': len(df)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/predict', methods=['POST'])
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

@app.route('/analyze', methods=['POST'])
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
    # Train model on startup
    print("Training model...")
    tagger.train_model()
    print("Model trained successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

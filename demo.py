#!/usr/bin/env python3
"""
Demo script to test the Blog Content Tagger functionality
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🚀 Testing Automated Blog Content Tagger API")
    print("=" * 50)
    
    # Test 1: Train Model
    print("\n1. Training the model...")
    try:
        response = requests.post(f"{base_url}/train", timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model training: {result['message']}")
        else:
            print(f"❌ Training failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the Flask app is running on localhost:5000")
        return
    except Exception as e:
        print(f"❌ Training error: {e}")
    
    # Test 2: Analyze Sample Content
    print("\n2. Analyzing sample blog content...")
    
    sample_content = {
        "title": "Getting Started with Machine Learning",
        "content": """
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve 
        from experience without being explicitly programmed. In this comprehensive guide, we'll explore the 
        fundamentals of machine learning, including supervised learning, unsupervised learning, and deep learning.
        
        Supervised learning involves training algorithms on labeled data to make predictions on new, unseen data.
        Common algorithms include linear regression, decision trees, and neural networks. Unsupervised learning,
        on the other hand, finds hidden patterns in data without labeled examples.
        
        Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and
        understand complex patterns in data. It has revolutionized fields like computer vision, natural language
        processing, and speech recognition.
        
        Whether you're a beginner programmer or an experienced developer, understanding machine learning
        concepts will be valuable for your career in technology and data science.
        """
    }
    
    try:
        response = requests.post(
            f"{base_url}/analyze",
            json=sample_content,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['analysis']
            
            print("✅ Analysis completed successfully!")
            print(f"\n📊 Content Statistics:")
            print(f"   • Words: {analysis['word_count']}")
            print(f"   • Characters: {analysis['char_count']}")
            print(f"   • Sentences: {analysis['sentence_count']}")
            print(f"   • Reading Time: {analysis['reading_time']} minutes")
            
            print(f"\n🏷️  Predicted Tags:")
            for tag, confidence in analysis['predicted_tags']:
                print(f"   • {tag}: {confidence*100:.1f}%")
            
            print(f"\n🔑 Top Keywords:")
            for keyword, count in analysis['top_keywords'][:5]:
                print(f"   • {keyword}: {count}")
                
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed! Check the web interface at http://localhost:5000")

if __name__ == "__main__":
    print("Starting demo in 3 seconds...")
    print("Make sure the Flask app is running (python app.py)")
    time.sleep(3)
    test_api()


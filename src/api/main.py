import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uvicorn
from datetime import datetime
from config.settings import settings

# Try to import AI responder, but don't fail if it doesn't exist
try:
    from src.models.ai_responder import AIResponder
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI Responder not available - using fallback responses")

# Initialize FastAPI app
app = FastAPI(
    title="TDS Virtual TA",
    description="A Virtual Teaching Assistant for Tools in Data Science course",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    context: str = "general"  # general, docker, git, python, etc.

class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: list
    timestamp: str
    enhanced: bool = False

# Global variable to store our knowledge base
knowledge_base = {}

@app.on_event("startup")
async def load_knowledge_base():
    """Load the scraped TDS course content on startup"""
    global knowledge_base
    
    try:
        # Load the combined TDS course data
        data_path = os.path.join(settings.RAW_DATA_PATH, 'tds_course_all.json')
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            print(f"✅ Loaded knowledge base with {len(knowledge_base)} sections")
            
            # Print summary of loaded content
            total_chars = sum(len(section.get('content', '')) for section in knowledge_base.values())
            print(f"📊 Total content: {total_chars:,} characters")
            
            for section_name, section_data in knowledge_base.items():
                content_length = len(section_data.get('content', ''))
                print(f"  📄 {section_name}: {content_length:,} characters")
        else:
            print(f"❌ Knowledge base file not found: {data_path}")
            knowledge_base = {}
            
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        knowledge_base = {}

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "TDS Virtual TA API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "knowledge_base_sections": len(knowledge_base),
        "ai_enhanced": AI_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_loaded": len(knowledge_base) > 0,
        "sections_available": list(knowledge_base.keys()) if knowledge_base else [],
        "ai_enhancement": AI_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/sections")
async def get_sections():
    """Get available knowledge base sections"""
    if not knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    sections_info = {}
    for section_name, section_data in knowledge_base.items():
        sections_info[section_name] = {
            "content_length": len(section_data.get('content', '')),
            "url": section_data.get('url', ''),
            "scraped_at": section_data.get('scraped_at', '')
        }
    
    return {
        "total_sections": len(knowledge_base),
        "sections": sections_info
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint to ask questions to the Virtual TA"""
    if not knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Search through the knowledge base and generate AI-enhanced responses
        answer_data = await search_knowledge_base(request.question, request.context)
        
        return AnswerResponse(
            answer=answer_data["answer"],
            confidence=answer_data["confidence"],
            sources=answer_data["sources"],
            enhanced=answer_data.get("enhanced", False),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

async def search_knowledge_base(question: str, context: str = "general"):
    """Search through the knowledge base and generate AI-enhanced responses"""
    
    # Simple keyword-based search
    question_lower = question.lower()
    relevant_sections = []
    
    # Search through all sections
    for section_name, section_data in knowledge_base.items():
        content = section_data.get('content', '').lower()
        
        # Simple relevance scoring based on keyword matches
        score = 0
        question_words = question_lower.split()
        
        for word in question_words:
            if len(word) > 2:  # Skip very short words
                score += content.count(word)
        
        if score > 0:
            relevant_sections.append({
                "section": section_name,
                "score": score,
                "content": section_data.get('content', ''),
                "url": section_data.get('url', '')
            })
    
    # Sort by relevance score
    relevant_sections.sort(key=lambda x: x['score'], reverse=True)
    
    if not relevant_sections:
        return {
            "answer": "I couldn't find specific information about your question in the TDS course materials. Could you please rephrase your question or ask about Docker, Git, or general TDS course topics?",
            "confidence": 0.1,
            "sources": [],
            "enhanced": False
        }
    
    # Take the most relevant section
    best_match = relevant_sections[0]
    sources = [f"{best_match['section']} section"]
    
    # Try to enhance with AI if available and AIPipe token is configured
    try:
        if (AI_AVAILABLE and 
            hasattr(settings, 'AIPIPE_TOKEN') and 
            settings.AIPIPE_TOKEN != "your_aipipe_token_here"):
            
            print("🤖 Attempting AI enhancement...")
            ai_responder = AIResponder()
            enhanced_response = ai_responder.generate_enhanced_response(
                question, 
                best_match['content'], 
                sources
            )
            print("✅ AI enhancement successful!")
            return enhanced_response
            
    except Exception as e:
        print(f"⚠️ AI enhancement failed, using fallback: {e}")
    
    # Fallback to simple response
    content = best_match['content']
    
    # Extract a relevant snippet (first 500 characters that contain keywords)
    snippet = ""
    question_words = question_lower.split()
    
    # Find the best snippet
    for i in range(0, len(content), 100):
        chunk = content[i:i+500]
        chunk_lower = chunk.lower()
        
        if any(word in chunk_lower for word in question_words if len(word) > 2):
            snippet = chunk
            break
    
    if not snippet:
        snippet = content[:500]
    
    return {
        "answer": f"Based on the TDS course materials:\n\n{snippet.strip()}...\n\nThis information comes from the {best_match['section']} section of the course.",
        "confidence": min(0.9, best_match['score'] / 10),  # Simple confidence calculation
        "sources": sources,
        "enhanced": False
    }

@app.get("/test-ai")
async def test_ai_integration():
    """Test endpoint to check AI integration status"""
    return {
        "ai_available": AI_AVAILABLE,
        "aipipe_configured": (hasattr(settings, 'AIPIPE_TOKEN') and 
                             settings.AIPIPE_TOKEN != "your_aipipe_token_here"),
        "knowledge_base_loaded": len(knowledge_base) > 0,
        "total_sections": len(knowledge_base)
    }

if __name__ == "__main__":
    print("🚀 Starting TDS Virtual TA API...")
    print(f"📊 Server will run on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"🤖 AI Enhancement: {'Available' if AI_AVAILABLE else 'Not Available'}")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )

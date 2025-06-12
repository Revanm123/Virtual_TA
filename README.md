# TDS Virtual Teaching Assistant

A Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras.

## Features
- ✅ Scrapes TDS course content (20,281 characters)
- ✅ FastAPI backend with intelligent search
- ✅ AI-enhanced responses (when AIPipe token provided)
- ✅ Web interface for easy interaction
- ✅ Comprehensive knowledge base

## Quick Start

1. **Start the API server**:
python -m src.api.main


2. **Open the web interface**:
- Open `web_interface.html` in your browser

3. **Ask questions** about TDS course content!

## API Endpoints
- `GET /` - Health check
- `GET /sections` - View available content sections
- `POST /ask` - Ask questions to the Virtual TA

## Project Structure
- `src/api/` - FastAPI backend
- `src/scraper/` - Web scrapers for TDS content
- `data/raw/` - Scraped course materials
- `config/` - Configuration settings
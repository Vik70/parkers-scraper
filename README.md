# Parkers Scraper

A comprehensive web scraping solution for extracting car review data from Parkers.co.uk.

## 🚀 Quick Start

**For the latest enhanced scraper system, see [README_ENHANCED_SCRAPER.md](README_ENHANCED_SCRAPER.md)**

The enhanced system includes:
- **Smart URL resolution** from broken review URLs to working ones
- **Structured content extraction** matching your JSON requirements
- **🆕 Smart Data Organization** - 22 organized columns instead of single JSON
- **Excel integration** with comprehensive data population
- **Batch processing** with concurrency control

## 📁 Project Structure

- **`enhanced_parkers_scraper.py`** - Main enhanced scraper engine
- **`resolver_clean.py`** - URL resolution logic
- **`test_small_batch.py`** - Test script for small batches
- **`run_full_scrape.py`** - Production script for full dataset
- **`Series + URL.xlsx`** - Input file with car data

## 🔧 Legacy Files

- `batch_parkers_scrape.py` - Original batch scraper
- `parkers_scrape_full.py` - Original full scraper
- `test_parkers_scraper.py` - Original test script

## 📊 What It Does

1. **Reads** your Excel file with car URLs
2. **Resolves** broken URLs by going to specs pages first
3. **Scrapes** review content from working URLs
4. **🆕 Organizes** data into 22 structured columns (Title, Price, Specs, Pros/Cons, etc.)
5. **Populates** Excel with organized, searchable data for easy analysis

## 🎯 Use Cases

- Extract car review data from Parkers.co.uk
- Handle broken/404 URLs automatically
- Generate structured JSON data for analysis
- Integrate with Excel workflows

## 📖 Documentation

- **[README_ENHANCED_SCRAPER.md](README_ENHANCED_SCRAPER.md)** - Detailed usage instructions, configuration options, and troubleshooting
- **[DATA_ORGANIZATION_GUIDE.md](DATA_ORGANIZATION_GUIDE.md)** - 🆕 Complete guide to the new 22-column data organization system

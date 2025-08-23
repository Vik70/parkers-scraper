# Enhanced Parkers Scraper

This enhanced scraper system solves the problem of broken review URLs by implementing a smart resolution strategy that goes to `/specs` pages first, then navigates to working `/review` pages.

## 🎯 What It Does

1. **Reads** your `Series + URL.xlsx` file (2736 car entries)
2. **Resolves** broken review URLs by going to specs pages first
3. **Scrapes** the actual review content from working URLs
4. **Formats** data as JSON matching your `parkers_article.json` structure
5. **Populates** column C in Excel with the scraped JSON data

## 🚀 Quick Start

### 1. Test with Small Batch
```bash
python test_small_batch.py
```
This processes just 5 URLs to verify everything works.

### 2. Run Full Production Scrape
```bash
python run_full_scrape.py
```
This processes all 2736 URLs (will take several hours).

## 📁 Files Overview

- **`enhanced_parkers_scraper.py`** - Main scraper engine
- **`resolver_clean.py`** - URL resolution logic (fixed version)
- **`test_small_batch.py`** - Test script for small batches
- **`run_full_scrape.py`** - Production script for full dataset
- **`Series + URL.xlsx`** - Your input file with car data

## 🔧 How It Works

### URL Resolution Strategy
1. **Input**: `https://www.parkers.co.uk/abarth/124-spider/convertible-2016/review/` (404 error)
2. **Convert to specs**: `https://www.parkers.co.uk/abarth/124-spider/convertible-2016/specs/`
3. **Find Review tab**: Locate the Review tab within the specs page
4. **Extract working URL**: Get the actual working review URL
5. **Scrape content**: Extract structured content from the working review page

### Content Extraction
The scraper extracts content in the same format as your `parkers_article.json`:
- **Title**: Car review title
- **Article text**: Main review content
- **Sections**: Structured sections like "Pros & cons", "Overview", "Verdict", etc.
- **URL**: Source URL for reference

## ⚙️ Configuration Options

### ScrapeConfig
```python
config = ScrapeConfig(
    headless=True,        # Run browser in background
    timeout_s=60,         # Page timeout in seconds
    save_debug_html=False, # Save HTML for debugging
    retry_count=2         # Retry failed scrapes
)
```

### Command Line Options
```bash
# Process specific number of rows
python enhanced_parkers_scraper.py --limit 10 --offset 0

# Set concurrency (browser instances)
python enhanced_parkers_scraper.py --concurrency 5

# Save debug HTML files
python enhanced_parkers_scraper.py --debug-html

# Custom output file
python enhanced_parkers_scraper.py --output my_results.xlsx
```

## 📊 Output Format

### Excel Structure
- **Column A**: Series (production years start-end)
- **Column B**: Model URL (original URLs from your file)
- **Column C**: JSON_Data (scraped content in JSON format)

### JSON Data Structure
```json
{
  "title": "Abarth 124 Spider Convertible (2016-2019) review",
  "article_text": "Main review content...",
  "article_markdown": "Same as article_text for now",
  "sections": {
    "full_text": "All page text",
    "At a glance": "Key specifications",
    "Pros & cons": "Advantages and disadvantages",
    "Overview": "General overview",
    "Verdict": "Final assessment",
    "Abarth 124 Spider Convertible rivals": "Competitor analysis"
  },
  "url": "https://www.parkers.co.uk/...",
  "processing_status": "success"
}
```

## 🚨 Important Notes

### Processing Time
- **Small batch (5 URLs)**: ~2-5 minutes
- **Full dataset (2736 URLs)**: 4-8 hours (depending on website response times)

### Concurrency
- **Recommended**: 3-5 concurrent browser instances
- **Too high**: May overwhelm the website or your system
- **Too low**: Will take much longer to complete

### Error Handling
- Failed URLs are marked with `"processing_status": "error"`
- The scraper continues processing other URLs
- A summary report shows success/failure rates

## 🔍 Troubleshooting

### Common Issues

1. **"Page not found" errors**
   - This is expected for direct review URLs
   - The resolver will automatically convert them to specs URLs

2. **Browser crashes**
   - Reduce concurrency (try 2-3 instead of 5)
   - Check system memory usage

3. **Slow performance**
   - Increase concurrency (up to 5)
   - Check internet connection speed

### Debug Mode
```bash
python enhanced_parkers_scraper.py --debug-html --headless false
```
This saves HTML files and shows browser windows for debugging.

## 📈 Progress Tracking

The scraper provides real-time progress updates:
- Current URL being processed
- Resolution status
- Scraping progress
- Final summary with success rates

## 🎉 Success Indicators

- **JSON data** appears in column C
- **Processing status** shows "success"
- **Sections** contain meaningful content
- **Article text** has substantial length (>1000 characters)

## 🔄 Resuming Interrupted Scrapes

If you need to stop and resume:
1. Note the last processed row number
2. Use `--offset` to start from where you left off
3. The scraper will continue from that point

## 📞 Support

If you encounter issues:
1. Check the debug HTML files (if enabled)
2. Verify the input Excel file format
3. Test with a small batch first
4. Check system resources (memory, disk space)

---

**Happy Scraping - By Vik the goat 🚗✨**

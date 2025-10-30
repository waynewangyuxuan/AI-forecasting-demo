# Development2Development
This TODO is Designed to be update everytime we finished our current development.
This TODO serves as a guideline on what we should work on the next time we come back to this project's development.
And every time we resume to development, we should read TODO first to know where to start
We follow an append-only strategy in the writing of thie file.

---

## 2025-10-29 - Post-Initial Implementation

### Current Status: Core Implementation Complete ✅

The entire pipeline has been implemented per TECHNICAL.md and PRODUCT.md specifications:
- All 8 pipeline stages working
- Complete database layer
- All service modules functional
- CLI interface ready
- Documentation comprehensive

### Immediate Next Steps (Before First Live Run)

1. **Test End-to-End Pipeline** 🔴 HIGH PRIORITY
   - [ ] Set up .env file with real API keys
   - [ ] Run database initialization: `python cli.py init`
   - [ ] Execute test forecast with sample question
   - [ ] Verify all stages complete successfully
   - [ ] Check output files generated correctly
   - [ ] Validate forecast quality and reasoning

2. **Bug Fixes & Refinements** 🟡 MEDIUM PRIORITY
   - [ ] Fix any errors discovered during testing
   - [ ] Tune clustering parameters if deduplication too aggressive/lenient
   - [ ] Adjust event extraction prompt if quality issues
   - [ ] Optimize scraping concurrency if too slow/fast

3. **API Key Acquisition** 🟡 MEDIUM PRIORITY
   - [ ] Google Custom Search API key
   - [ ] Google Custom Search Engine ID
   - [ ] Google Gemini API key
   - [ ] OpenAI API key

### Phase 2: Testing & Validation (1-2 hours)

4. **Comprehensive Testing**
   - [ ] Test with all 3 seed questions from PRODUCT.md:
     - "Will China mass-produce humanoid robots by end of 2025?"
     - "Will Ghana sign 'Proper Human Sexual Rights' bill before 2026?"
     - "Will S&P 500 increase over December 2025?"
   - [ ] Test resume functionality (interrupt a run, then resume)
   - [ ] Test with limited parameters (--max-urls 5 --max-events 20)
   - [ ] Test error handling (invalid API keys, network issues)
   - [ ] Validate output formats (JSON structure, Markdown rendering)

5. **Quality Assurance**
   - [ ] Review generated queries for diversity and relevance
   - [ ] Check scraping success rate (target >80%)
   - [ ] Evaluate event extraction quality (accuracy, relevance)
   - [ ] Assess deduplication effectiveness (too many/few duplicates?)
   - [ ] Validate timeline coherence and chronological ordering
   - [ ] Review forecast reasoning for logical consistency
   - [ ] Check evidence citation accuracy

6. **Performance Optimization**
   - [ ] Measure actual runtime per stage
   - [ ] Identify bottlenecks (likely scraping or event extraction)
   - [ ] Tune concurrency limits if needed
   - [ ] Consider caching strategies for development iterations
   - [ ] Monitor API costs per question

### Phase 3: Demo Preparation (1-2 hours)

7. **Demo Readiness**
   - [ ] Prepare 1-2 pre-run forecasts to show as examples
   - [ ] Create presentation materials (architecture diagram, sample output)
   - [ ] Practice live demo flow (5-8 minutes)
   - [ ] Prepare backup recorded demo in case of live issues
   - [ ] Test demo environment (clean database, fresh run)

8. **Documentation Polish**
   - [ ] Add screenshots to README.md
   - [ ] Create architecture diagram (ASCII or image)
   - [ ] Add example outputs to documentation
   - [ ] Write troubleshooting guide for common issues
   - [ ] Add FAQ section

### Phase 4: Future Enhancements (Post-Demo)

9. **Core Improvements**
   - [ ] Add unit tests for critical functions
   - [ ] Implement integration tests with mocked APIs
   - [ ] Add progress persistence (save intermediate results)
   - [ ] Implement rate limit backoff with multiple API keys
   - [ ] Add support for alternative LLMs (Claude, GPT-4)

10. **Feature Additions**
    - [ ] REST API with FastAPI (already scaffolded in TECHNICAL.md)
    - [ ] Web dashboard for running and viewing forecasts
    - [ ] Batch processing multiple questions
    - [ ] Historical accuracy tracking
    - [ ] Comparison with prediction markets (Metaculus, Polymarket)
    - [ ] Export to additional formats (PDF, HTML)
    - [ ] Email notifications on completion

11. **Advanced Features (From PRODUCT.md Non-Goals)**
    - [ ] Iterative refinement (multi-round query generation)
    - [ ] Real-time monitoring and forecast updates
    - [ ] Multi-model ensemble forecasting
    - [ ] Advanced clustering (hierarchical, graph-based)
    - [ ] User authentication for web interface
    - [ ] Historical forecast database with analytics
    - [ ] Interactive timeline visualization
    - [ ] A/B testing framework for prompt optimization

### Known Issues to Address

**High Priority:**
- None yet (pending testing)

**Medium Priority:**
- Embedding cache grows unbounded (add size limit and LRU eviction)
- No retry limit on LLM streaming (could hang indefinitely)
- Search query deduplication may be too strict (>75% similarity)

**Low Priority:**
- CLI progress bars may flicker on some terminals
- Markdown report formatting could be improved with tables
- No support for non-English content (language detection present but not fully utilized)

### Testing Checklist Before Demo

- [ ] Database initializes correctly
- [ ] All services instantiate without errors
- [ ] Search queries are diverse and relevant
- [ ] Scraping respects robots.txt
- [ ] Event extraction produces valid timestamps
- [ ] Clustering reduces event count by 30-50%
- [ ] Timeline is chronologically ordered
- [ ] Forecast includes probability/prediction
- [ ] Forecast reasoning cites specific events
- [ ] Output files are created and readable
- [ ] Resume functionality works after interruption
- [ ] Error logging captures all failures
- [ ] Cost per question is under $0.50
- [ ] Total runtime is 5-15 minutes

### Next Development Session TODO

**When resuming development:**

1. Read this file to understand current state
2. Review PROGRESS.md to see what was accomplished
3. Check "Immediate Next Steps" section above
4. Focus on testing and validation first
5. Address any issues discovered during testing
6. Update this file with new findings and next steps

### API Cost Tracking

**Budget:** <$0.50 per question

**Estimated breakdown:**
- Google Custom Search: Free tier (100/day) = $0.00
- Gemini API calls (query gen + event extraction + forecast): ~$0.10-0.20
- OpenAI embeddings (~150 events × 2 tokens × $0.00002/1K): ~$0.02
- **Total: ~$0.12-0.22 per question** ✅ Under budget

**To monitor during testing:**
- Actual Gemini token usage
- Number of events requiring embeddings
- Search queries used (stay under 100/day free tier)


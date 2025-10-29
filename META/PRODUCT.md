# PRODUCT.md
## RAG-Based Forecasting Pipeline - Product Design Document

---

## 1. Executive Summary

### 1.1 Product Vision
A Retrieval-Augmented Generation (RAG) system that combines real-time web information with large language model reasoning to produce evidence-based forecasts for complex prediction questions.

### 1.2 Core Value Proposition
- **Automated Intelligence Gathering**: Replaces manual research with automated web scraping and information synthesis
- **Structured Knowledge Processing**: Transforms unstructured web content into organized event timelines
- **Explainable Predictions**: Provides transparent reasoning chains alongside probability estimates
- **Rapid Deployment**: Designed for 3-6 hour implementation timeframe

### 1.3 Target Users
- Researchers exploring LLM-based forecasting methodologies
- Data scientists evaluating RAG pipeline architectures
- Trial project participants demonstrating technical capabilities

---

## 2. Product Objectives

### 2.1 Primary Goals
1. **Demonstrate Rapid Iteration Capability**: Complete end-to-end pipeline in 3-6 hours
2. **Showcase Engineering Proficiency**: Integrate multiple APIs, handle data processing, implement web scraping
3. **Apply Research Concepts**: Implement simplified versions of LLM-TLS and Chronos methodologies
4. **Generate Actionable Forecasts**: Produce probability estimates with supporting evidence

### 2.2 Success Metrics
- Successfully process all 3 provided forecasting questions
- Generate ~100 relevant URLs per question
- Extract and deduplicate events with >70% relevance
- Produce forecasts with complete reasoning traces
- Complete live demo with full logging visibility

---

## 3. User Journey

### 3.1 High-Level User Flow
```
Input Question → Query Generation → Web Search → Content Extraction → 
Event Processing → Timeline Construction → Forecast Generation → Output
```

### 3.2 Detailed User Experience

#### Phase 1: Question Input
- **User Action**: Provide forecasting question (e.g., "Will China mass-produce humanoid robots by end of 2025?")
- **System Response**: Acknowledge question and begin processing
- **Expected Duration**: < 1 second

#### Phase 2: Information Gathering (Automated)
- **System Actions**:
  - Generate 10 optimized search queries using LLM
  - Execute Google searches via API
  - Retrieve top 10 URLs per query (~100 total)
  - Scrape textual content from all URLs
- **User Visibility**: Progress logging with query/URL counts
- **Expected Duration**: 2-5 minutes

#### Phase 3: Knowledge Processing (Automated)
- **System Actions**:
  - Extract events with timestamps from scraped content
  - Generate embeddings for all events
  - Cluster and deduplicate similar events
  - Sort chronologically into timeline
- **User Visibility**: Event extraction statistics, deduplication metrics
- **Expected Duration**: 3-7 minutes

#### Phase 4: Forecast Generation (Automated)
- **System Actions**:
  - Feed timeline + original question to LLM
  - Generate structured forecast with reasoning
  - Format output with probability/prediction
- **User Visibility**: Full reasoning trace displayed
- **Expected Duration**: 30-60 seconds

#### Phase 5: Results Review
- **User Action**: Review complete output
- **Output Includes**:
  - Original question
  - Generated search queries
  - Event timeline (chronological)
  - Reasoning chain
  - Final forecast (probability or binary prediction)
  - Confidence level and caveats

---

## 4. Core Features

### 4.1 Query Generation Engine
**Purpose**: Generate diverse, relevant search queries for comprehensive information retrieval

**Capabilities**:
- LLM-powered query formulation (Gemini 2.5 Flash)
- Context-aware query diversification
- Temporal specificity (e.g., "2025", "recent developments")
- Multi-angle coverage (technical, market, regulatory, geopolitical)

**Input**: Single forecasting question  
**Output**: ~10 optimized Google search queries  
**Quality Criteria**: Queries should minimize overlap while maximizing information coverage

---

### 4.2 Web Search & Scraping Module
**Purpose**: Collect raw information from the internet

**Capabilities**:
- Google Custom Search JSON API integration
- Top-10 URL extraction per query
- Robust web scraping with error handling
- Content cleaning (remove ads, navigation, footers)
- Respect for robots.txt and rate limiting

**Input**: 10 search queries  
**Output**: ~100 URLs with extracted main text content  
**Quality Criteria**: >80% successful scrapes, text length >200 chars per article

---

### 4.3 Event Extraction Engine
**Purpose**: Transform unstructured text into structured event records

**Capabilities**:
- LLM-based event identification
- Timestamp extraction and normalization
- Event description summarization
- Relevance filtering

**Input**: Scraped article text  
**Output**: List of events with structure:
```
{
  "timestamp": "2024-11-15",
  "description": "Chinese robotics firm unveils first mass-production line",
  "source_url": "https://...",
  "relevance_score": 0.92
}
```
**Quality Criteria**: Each event must have valid timestamp and coherent description

---

### 4.4 Event Deduplication System
**Purpose**: Consolidate redundant information using semantic similarity

**Capabilities**:
- OpenAI text-embedding-small for vectorization
- Clustering algorithms (K-Means or DBSCAN)
- Intelligent event merging
- Source attribution preservation

**Input**: N raw events  
**Output**: M deduplicated events (where M < N)  
**Quality Criteria**: Maintain unique information while removing ~30-50% duplicates

---

### 4.5 Timeline Constructor
**Purpose**: Organize events into chronological narrative

**Capabilities**:
- Temporal sorting
- Gap identification
- Milestone highlighting
- Visual timeline formatting (text-based)

**Input**: Deduplicated events  
**Output**: Chronologically ordered timeline with clear structure  
**Quality Criteria**: Accurate temporal ordering, logical event sequencing

---

### 4.6 Forecasting Engine
**Purpose**: Generate evidence-based predictions with transparent reasoning

**Capabilities**:
- LLM-based analysis (Gemini 2.5 Flash)
- Timeline-aware reasoning
- Probability estimation or binary prediction
- Uncertainty quantification
- Supporting evidence citation

**Input**: Original question + Event timeline  
**Output**: Structured forecast containing:
```
{
  "question": "Will China mass-produce humanoid robots by end of 2025?",
  "prediction": "60% probability",
  "reasoning": [
    "Historical trend analysis...",
    "Current development stage...",
    "Regulatory environment..."
  ],
  "key_evidence": [Event IDs],
  "confidence": "Medium",
  "caveats": ["Limited visibility into proprietary developments", ...]
}
```

---

## 5. User Interface Requirements

### 5.1 CLI-Based Interface (Primary)
**Rationale**: Fastest to implement for 3-6 hour constraint

**Features**:
- Command to run forecast: `python forecast.py --question "..." --output results/`
- Real-time progress logging with timestamps
- Color-coded status messages (info/success/warning/error)
- Structured JSON output file
- Human-readable summary to terminal

**Example Output**:
```
[2025-10-29 14:23:01] INFO: Starting forecast pipeline
[2025-10-29 14:23:02] INFO: Generating search queries...
[2025-10-29 14:23:08] SUCCESS: Generated 10 queries
[2025-10-29 14:23:09] INFO: Executing Google searches...
[2025-10-29 14:25:34] SUCCESS: Retrieved 97 URLs
[2025-10-29 14:25:35] INFO: Scraping web content...
[2025-10-29 14:28:12] SUCCESS: Scraped 94/97 URLs successfully
[2025-10-29 14:28:13] INFO: Extracting events...
[2025-10-29 14:32:45] SUCCESS: Extracted 342 events
[2025-10-29 14:32:46] INFO: Deduplicating events...
[2025-10-29 14:33:21] SUCCESS: Reduced to 156 unique events
[2025-10-29 14:33:22] INFO: Generating forecast...
[2025-10-29 14:34:18] SUCCESS: Forecast complete!

========================================
FORECAST RESULTS
========================================
Question: Will China be able to mass-produce humanoid robots by the end of 2025?
Prediction: 55% probability
Confidence: Medium

Key Reasoning:
1. Major manufacturers (BYD, Xiaomi) announced 2025 production targets
2. Current prototypes in testing phase as of Q3 2024
3. Supply chain capacity questions remain unresolved
...

Full results saved to: results/forecast_20251029_142318.json
```

---

### 5.2 Web Dashboard (Optional Enhancement)
**Rationale**: Better for live demo presentation

**Core Screens**:
1. **Input Screen**: Text area for question, submit button
2. **Processing Screen**: Real-time progress visualization, loading animations
3. **Results Screen**: 
   - Timeline visualization (interactive)
   - Reasoning breakdown
   - Prediction display with confidence meters
   - Export options (JSON, PDF)

**Technology Suggestions**:
- Frontend: React or simple Flask templates
- Visualization: D3.js for timeline, Chart.js for probabilities
- Real-time updates: WebSockets or SSE

---

## 6. Data Models

### 6.1 Forecasting Question
```python
{
  "id": "q001",
  "text": "Will China be able to mass-produce humanoid robots by the end of 2025?",
  "category": "technology",
  "resolution_date": "2025-12-31",
  "created_at": "2025-10-29T14:23:01Z"
}
```

### 6.2 Search Query
```python
{
  "query_id": "sq001",
  "question_id": "q001",
  "text": "China humanoid robot mass production 2025",
  "created_at": "2025-10-29T14:23:05Z"
}
```

### 6.3 Scraped Article
```python
{
  "article_id": "art001",
  "url": "https://example.com/article",
  "query_id": "sq001",
  "title": "Chinese Robotics Firms Race to Production",
  "content": "Full article text...",
  "scraped_at": "2025-10-29T14:25:47Z",
  "status": "success",
  "word_count": 1250
}
```

### 6.4 Event
```python
{
  "event_id": "evt001",
  "timestamp": "2024-11-15",
  "description": "BYD announces humanoid robot production line in Shenzhen",
  "source_article_id": "art001",
  "source_url": "https://example.com/article",
  "embedding": [0.023, -0.145, ...],  # 1536-dim vector
  "cluster_id": 5,
  "relevance_score": 0.89
}
```

### 6.5 Timeline
```python
{
  "timeline_id": "tl001",
  "question_id": "q001",
  "events": ["evt012", "evt034", "evt045", ...],  # Ordered by timestamp
  "created_at": "2025-10-29T14:33:22Z",
  "event_count": 156
}
```

### 6.6 Forecast
```python
{
  "forecast_id": "fc001",
  "question_id": "q001",
  "timeline_id": "tl001",
  "prediction": "55% probability",
  "prediction_type": "probability",  # or "binary", "numeric"
  "reasoning_steps": [
    "Analysis of recent manufacturing announcements...",
    "Assessment of technological readiness...",
    "Evaluation of regulatory environment..."
  ],
  "key_evidence_events": ["evt012", "evt034", "evt089"],
  "confidence_level": "medium",
  "caveats": [
    "Limited visibility into classified developments",
    "Supply chain volatility not fully modeled"
  ],
  "generated_at": "2025-10-29T14:34:18Z"
}
```

---

## 7. Quality & Performance Requirements

### 7.1 Accuracy Targets
- **Search Relevance**: >75% of URLs should be relevant to question
- **Event Extraction**: >80% of extracted events should be factually accurate
- **Timeline Coherence**: Events should follow logical chronological flow
- **Forecast Calibration**: Probabilities should reflect actual uncertainty

### 7.2 Performance Targets
- **Total Pipeline Duration**: 5-15 minutes per question
- **Query Generation**: <10 seconds
- **Web Scraping**: <5 minutes for 100 URLs
- **Event Processing**: <5 minutes for 100 articles
- **Forecast Generation**: <60 seconds

### 7.3 Reliability Requirements
- **Scraping Success Rate**: >80% of URLs successfully scraped
- **API Resilience**: Graceful handling of rate limits and timeouts
- **Error Recovery**: Continue processing if individual components fail
- **Logging**: Complete audit trail of all operations

### 7.4 Cost Constraints
- **Total Cost per Question**: <$0.50 USD
  - Gemini API: Free tier usage
  - OpenAI Embeddings: ~$0.02 per question (156 events × 2 tokens avg × $0.00002/1K)
  - Google Search API: Free tier (100 queries/day)

---

## 8. Compliance & Constraints

### 8.1 API Usage Limits
- **Google Custom Search**: 100 queries/day per API key (mitigation: multiple accounts)
- **Gemini API**: Free tier limits vary (monitor usage)
- **OpenAI API**: Pay-as-you-go (minimal cost expected)

### 8.2 Ethical Considerations
- **Web Scraping**: Respect robots.txt, implement rate limiting
- **Attribution**: Maintain source URLs in all extracted events
- **Bias Awareness**: Acknowledge that forecasts reflect web content biases
- **Transparency**: Provide full reasoning traces for auditability

### 8.3 Technical Constraints
- **Time Limit**: Entire project must be completable in 3-6 hours
- **Simplicity**: Avoid over-engineering; prioritize working prototype
- **Dependencies**: Minimize external dependencies for easier setup

---

## 9. Non-Goals (Out of Scope)

### 9.1 Explicitly Excluded Features
- ❌ **Iterative Refinement**: No multi-round query regeneration (unlike full LLM-TLS)
- ❌ **Real-Time Updates**: No continuous monitoring or forecast updates
- ❌ **Multi-Model Ensemble**: Single LLM model only (Gemini)
- ❌ **Advanced Clustering**: Simple K-Means/DBSCAN only, no hierarchical methods
- ❌ **User Authentication**: No login system required
- ❌ **Historical Data Storage**: No persistent database (file-based output only)
- ❌ **Advanced Visualization**: No complex interactive graphs (text-based timeline acceptable)
- ❌ **Production Deployment**: Demo-quality code, not production-ready
- ❌ **A/B Testing**: No experimentation framework
- ❌ **Mobile Support**: Desktop-only experience

### 9.2 Future Enhancements (Post-Trial)
- Multi-round iterative forecasting
- Comparison with prediction markets (e.g., Metaculus, Polymarket)
- Automated question generation from news feeds
- Collaborative forecasting with multiple users
- Historical accuracy tracking and calibration scoring

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Google Search API quota exhaustion | High | Medium | Use multiple API keys, implement caching |
| Web scraping failures (403, 429 errors) | Medium | High | Retry logic, user-agent rotation, skip failed URLs |
| LLM hallucination in event extraction | High | Medium | Cross-reference with source text, implement validation |
| Embedding cost exceeds budget | Low | Low | Monitor token usage, optimize event descriptions |
| Timeline incoherence due to poor deduplication | Medium | Medium | Tune clustering parameters, manual review of sample |

### 10.2 Execution Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Cannot complete in 3-6 hour timeframe | High | Low | Prioritize MVP features, skip optional enhancements |
| Insufficient test data for validation | Medium | Low | Use provided 3 questions thoroughly |
| Demo environment issues during presentation | High | Low | Test in clean environment, prepare backup recordings |

---

## 11. Success Criteria for Live Demo

### 11.1 Must-Have Demonstrations
1. ✅ **End-to-End Execution**: Run complete pipeline on at least 1 question
2. ✅ **Logging Visibility**: Show real-time progress logs during execution
3. ✅ **Timeline Output**: Display structured, chronological event timeline
4. ✅ **Reasoning Trace**: Show complete reasoning steps from LLM
5. ✅ **Final Prediction**: Present clear forecast with probability/prediction

### 11.2 Nice-to-Have Demonstrations
- Run on all 3 provided questions
- Compare outputs across different questions
- Show deduplication effectiveness (before/after event counts)
- Display sample scraped content vs. extracted events
- Demonstrate error handling with problematic URL

### 11.3 Demo Presentation Structure (15-20 minutes)
1. **Introduction** (2 min): Explain problem and approach
2. **Architecture Overview** (3 min): High-level system diagram
3. **Live Execution** (8 min): Run pipeline on 1 question with commentary
4. **Results Analysis** (5 min): Walk through output, discuss quality
5. **Q&A** (2 min): Answer technical questions

---

## 12. Development Phases

### 12.1 Phase 1: Foundation (1-1.5 hours)
- Set up project structure
- Configure API keys (Gemini, OpenAI, Google Search)
- Implement query generation module
- Test with 1 sample question

### 12.2 Phase 2: Data Collection (1-1.5 hours)
- Implement Google Search API integration
- Build web scraping module with error handling
- Test on 10-20 URLs
- Validate content extraction quality

### 12.3 Phase 3: Knowledge Processing (1.5-2 hours)
- Implement event extraction with LLM
- Build embedding and clustering pipeline
- Create timeline construction logic
- Test deduplication effectiveness

### 12.4 Phase 4: Forecasting (0.5-1 hour)
- Implement forecast generation with LLM
- Structure output format
- Test reasoning quality

### 12.5 Phase 5: Integration & Testing (0.5-1 hour)
- Connect all modules end-to-end
- Run on all 3 questions
- Fix bugs and edge cases
- Prepare demo environment

---

## 13. Open Questions & Decisions Needed

### 13.1 Technical Decisions
- **Clustering Algorithm**: K-Means (simpler) vs. DBSCAN (density-based)?
- **Embedding Dimension**: Use full 1536-dim or reduce with PCA?
- **Event Relevance Filtering**: Hard threshold or soft ranking?
- **Forecast Format**: Probability percentage, confidence intervals, or binary?

### 13.2 Product Decisions
- **Output Format**: JSON only, or also generate PDF/HTML report?
- **Intermediate Results**: Save to disk for debugging, or memory-only?
- **Question Input**: CLI argument, config file, or interactive prompt?

---

## 14. Appendix

### 14.1 Example Forecasting Questions

**Question 1: Technology/Business**
> "Will China be able to mass-produce humanoid robots by the end of 2025?"
- **Resolution Criteria**: Evidence of >10,000 units/year production capacity
- **Key Information Sources**: Manufacturing announcements, industry reports, tech news

**Question 2: Policy/Legal**
> "Will Ghana sign into law 'The Proper Human Sexual Rights and Ghanaian Family Values' bill before 2026?"
- **Resolution Criteria**: Official government signing and publication
- **Key Information Sources**: Ghana government websites, international news, policy trackers

**Question 3: Financial/Economic**
> "Will the S&P 500 Index increase over December 2025?"
- **Resolution Criteria**: Index value on Dec 31, 2025 > Nov 30, 2025
- **Key Information Sources**: Market analysis, economic indicators, Fed announcements

### 14.2 Reference Papers Summary

**LLM-TLS (Timeline Summarization)**
- Core Idea: Incremental timeline construction with deduplication
- Key Technique: Embedding-based event clustering
- Adaptation: Simplified single-pass instead of iterative refinement

**Chronos (News Retrieval)**
- Core Idea: Self-questioning for better query generation
- Key Technique: LLM generates follow-up questions for comprehensive coverage
- Adaptation: One-shot query generation with diversity prompting

---

## 15. Glossary

- **RAG (Retrieval-Augmented Generation)**: Pattern combining information retrieval with LLM generation
- **Event Timeline**: Chronologically ordered sequence of extracted events
- **Deduplication**: Process of identifying and merging similar/redundant events
- **Embedding**: Dense vector representation of text for semantic similarity
- **Clustering**: Grouping similar items based on distance metrics
- **Forecast**: Prediction with probability estimate and supporting reasoning
- **Query Generation**: Creating search terms optimized for information retrieval

---

**Document Version**: 1.0  
**Last Updated**: October 29, 2025  
**Status**: Ready for Technical Design Phase
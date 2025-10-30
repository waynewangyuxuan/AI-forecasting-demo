-- AI Forecasting Pipeline Database Schema
-- SQLite Database Schema for tracking forecasting runs and results
-- Version: 1.0.0

-- Questions table: Stores the forecasting questions to be answered
CREATE TABLE IF NOT EXISTS Questions (
  id INTEGER PRIMARY KEY,
  question_text TEXT NOT NULL,
  resolution_criteria TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Runs table: Tracks each forecasting pipeline execution
CREATE TABLE IF NOT EXISTS Runs (
  id INTEGER PRIMARY KEY,
  question_id INTEGER NOT NULL,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT,
  status TEXT CHECK(status IN ('PENDING','RUNNING','FAILED','COMPLETED')),
  stage_state TEXT,
  git_commit TEXT,
  FOREIGN KEY(question_id) REFERENCES Questions(id)
);

-- SearchQueries table: Stores generated search queries for each run
CREATE TABLE IF NOT EXISTS SearchQueries (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  query_text TEXT NOT NULL,
  prompt_version TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(run_id, query_text),
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

-- SearchResults table: Stores raw search results from search engines
CREATE TABLE IF NOT EXISTS SearchResults (
  id INTEGER PRIMARY KEY,
  query_id INTEGER NOT NULL,
  url TEXT NOT NULL,
  title TEXT,
  snippet TEXT,
  rank INTEGER,
  UNIQUE(query_id, url),
  FOREIGN KEY(query_id) REFERENCES SearchQueries(id)
);

-- Documents table: Stores fetched and cleaned document content
CREATE TABLE IF NOT EXISTS Documents (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  url TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  raw_content TEXT,
  cleaned_content TEXT,
  fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
  status TEXT,
  UNIQUE(run_id, content_hash),
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

-- Events table: Stores extracted events from documents
CREATE TABLE IF NOT EXISTS Events (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  event_time TEXT,
  headline TEXT,
  body TEXT,
  actors TEXT,
  confidence REAL,
  raw_response TEXT,
  FOREIGN KEY(document_id) REFERENCES Documents(id)
);

-- Embeddings table: Stores vector embeddings for events
CREATE TABLE IF NOT EXISTS Embeddings (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL,
  vector BLOB NOT NULL,
  model TEXT,
  dimensions INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(event_id, model),
  FOREIGN KEY(event_id) REFERENCES Events(id)
);

-- EventClusters table: Stores clusters of similar events
CREATE TABLE IF NOT EXISTS EventClusters (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  label TEXT,
  centroid_event_id INTEGER,
  member_ids TEXT,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

-- Timeline table: Stores the synthesized timeline of events
CREATE TABLE IF NOT EXISTS Timeline (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  cluster_id INTEGER,
  event_time TEXT,
  summary TEXT,
  citations TEXT,
  tags TEXT,
  FOREIGN KEY(run_id) REFERENCES Runs(id),
  FOREIGN KEY(cluster_id) REFERENCES EventClusters(id)
);

-- Forecasts table: Stores the final forecast predictions
CREATE TABLE IF NOT EXISTS Forecasts (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  probability REAL,
  reasoning TEXT,
  caveats TEXT,
  raw_response TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

-- RunMetrics table: Stores performance metrics for each run
CREATE TABLE IF NOT EXISTS RunMetrics (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  metric_name TEXT,
  metric_value REAL,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

-- Errors table: Stores errors that occur during pipeline execution
CREATE TABLE IF NOT EXISTS Errors (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  stage TEXT,
  reference TEXT,
  error_type TEXT,
  message TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

-- Indices for frequently queried columns
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON Documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_events_event_time ON Events(event_time);
CREATE INDEX IF NOT EXISTS idx_timeline_run_id ON Timeline(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON Runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_question_id ON Runs(question_id);
CREATE INDEX IF NOT EXISTS idx_search_queries_run_id ON SearchQueries(run_id);
CREATE INDEX IF NOT EXISTS idx_documents_run_id ON Documents(run_id);
CREATE INDEX IF NOT EXISTS idx_events_document_id ON Events(document_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_run_id ON Forecasts(run_id);
CREATE INDEX IF NOT EXISTS idx_errors_run_id ON Errors(run_id);

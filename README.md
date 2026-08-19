RAG Production

A production-oriented Retrieval-Augmented Generation (RAG) application built with Python and FastAPI.

The system answers questions using information retrieved from user-provided documents. It is designed as a closed-world RAG system: the answer must be supported by the documents available to the application. If the available documents do not provide enough evidence, the system should not rely on outside knowledge.

The project focuses not only on building a RAG pipeline, but also on the engineering required to make it reliable and maintainable in a production environment.

Features

- Document ingestion and recursive directory loading
- Document chunking and processing
- Vector retrieval using Qdrant
- Sentence-transformer embeddings
- Cross-encoder reranking
- LLM-based answer generation
- NLI-based grounding verification
- Source citations in responses
- Configurable retrieval limits and safety constraints
- Structured JSON logging
- Request IDs and request-duration monitoring
- Centralized exception handling
- API validation and hardening
- Health-check endpoint
- Docker containerization
- Automated tests
- GitHub Actions CI pipeline
- Automated Docker image build validation

Architecture

The high-level request flow is:

                    ┌──────────────────┐
                    │    FastAPI API   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   RAG Pipeline   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Query Validation  Retrieval     Generation
                             │              │
                             ▼              ▼
                         Qdrant         LLM
                             │
                             ▼
                         Reranking
                             │
                             ▼
                    Grounding Verification
                             │
                             ▼
                       Final Response

For document ingestion:

Documents
    │
    ▼
Document Loader
    │
    ▼
Text Processing / Chunking
    │
    ▼
Embeddings
    │
    ▼
Qdrant Vector Store

Technology Stack

Backend

- Python 3.12
- FastAPI
- Pydantic
- Pydantic Settings
- Uvicorn

Retrieval

- Qdrant
- Sentence Transformers
- BGE embeddings
- Cross-encoder reranking

Generation and Grounding

- Hugging Face Transformers
- Qwen
- NLI-based verification

Engineering

- Pytest
- uv
- Docker
- GitHub Actions
- Structured logging

API

The main API endpoint is:

POST /api/v1/query

Example request:

{
  "query": "What does the document say about ...?",
  "retrieval_limit": 5
}

The response contains the generated answer, grounding information, and the document sources used to produce the answer.

Health Check

GET /health

Example:

{
  "status": "ok",
  "environment": "development"
}

Configuration

Application configuration is handled through Pydantic Settings.

Environment variables use the "RAG_" prefix.

Examples:

RAG_ENVIRONMENT=production
RAG_LOG_LEVEL=INFO
RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAG_QDRANT_PATH=data/qdrant
RAG_RETRIEVAL_K=20
RAG_MAX_RETRIEVAL_LIMIT=20

Configuration values are validated when the application starts. Limits such as embedding dimensions, document sizes, query length, and retrieval limits cannot be set to invalid values.

Local Development

Clone the repository:

git clone https://github.com/rahmandarji/rag-production.git
cd rag-production

Create and activate the environment:

uv sync
source .venv/bin/activate

Run the application:

uv run uvicorn app.main:app --reload

The API will be available at:

http://localhost:8000

Health check:

curl http://localhost:8000/health

Running Tests

The project contains tests covering configuration, ingestion, retrieval, grounding, API behavior, and production hardening.

Run the full test suite:

uv run pytest -q

The final local test suite passed with:

122 passed, 7 warnings

The remaining warnings are related to a deprecation inside the PyTorch stack rather than failures in the application tests.

Docker

The application can be built as a Docker image:

docker build -t rag-production:latest .

Run the container:

docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/data/qdrant:/app/data/qdrant" \
  rag-production:latest

Then check:

curl http://localhost:8000/health

The Docker image uses a non-root application user and runs the FastAPI application through Uvicorn.

Observability

The application uses structured JSON logging rather than plain text logs.

Example events include:

request_started
request_completed
internal_server_error

Request logs contain useful operational information such as:

- Request ID
- HTTP method
- Endpoint
- Status code
- Request duration
- Log level
- Logger name

Each request also receives an "X-Request-ID" response header.

This makes it easier to trace a request through application logs.

API Hardening

The API validates incoming requests and protects application boundaries with limits such as:

- Maximum query length
- Retrieval-limit validation
- Document-size limits
- Batch-size limits
- Maximum chunks per document
- Valid configuration environments
- Valid log levels

Unexpected internal exceptions are handled centrally so internal implementation details are not returned to API clients.

CI/CD

GitHub Actions is used to automatically validate changes.

The CI pipeline currently performs:

1. Repository checkout
2. Python 3.12 setup
3. uv installation
4. Dependency installation
5. Full test execution
6. Docker image build

The Docker build runs only after the test job succeeds.

This prevents a broken test suite from being treated as a valid production image.

Production Engineering Work

This project was developed beyond the basic "build a RAG demo" stage.

The production work covered:

RAG Pipeline
     ↓
Configuration Validation
     ↓
Testing
     ↓
Observability & Monitoring
     ↓
API / Service Hardening
     ↓
Containerization
     ↓
CI/CD
     ↓
Production Deployment Preparation
     ↓
Load & Performance Testing
     ↓
Final Production Validation

The goal was to understand the engineering decisions involved in taking a GenAI application from a working prototype toward a production service.

Project Structure

A simplified project structure:

rag-production/
│
├── app/
│   ├── api/
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── container.py
│   │   └── logging.py
│   │
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── grounding/
│   ├── pipeline/
│   └── main.py
│
├── tests/
│   ├── api/
│   ├── core/
│   ├── evaluation/
│   ├── grounding/
│   └── ingestion/
│
├── data/
│   └── qdrant/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md

Design Principle

The main design principle of this project is grounded answers over plausible answers.

The LLM is responsible for interpreting and generating an answer from retrieved evidence. It is not treated as the source of truth.

The retrieval and grounding layers provide the evidence boundary:

User Documents
      ↓
Retrieval
      ↓
Evidence
      ↓
LLM
      ↓
Grounding Verification
      ↓
Answer + Sources

When the available documents are insufficient, the system should communicate that limitation instead of presenting unsupported information as fact.

Current Status

The project has completed its production-engineering workflow:

- RAG pipeline implemented
- Automated test suite passing
- Configuration validation implemented
- Observability implemented
- API hardening implemented
- Docker image builds successfully
- CI pipeline implemented
- GitHub Actions validation passing
- Final code pushed to GitHub

The project is intended primarily as a learning and engineering portfolio project demonstrating the transition from a RAG prototype to a more production-oriented GenAI service.

Future Improvements

Possible future work includes:

- Deploying the service to a cloud platform
- Adding authentication and authorization
- Externalizing vector storage
- Adding metrics collection and dashboards
- Distributed tracing
- Load testing against realistic workloads
- Model-serving optimization
- Horizontal scaling
- Automated deployment after CI
- More extensive evaluation datasets

License

This project is currently intended as a personal engineering and learning project.

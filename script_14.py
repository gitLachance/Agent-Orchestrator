# Create sample configuration files

# Development configuration
dev_config = '''# Development Configuration for Legal Agent Orchestrator

# LLM Provider Settings
llm_providers:
  openai:
    enabled: true
    default_model: "gpt-4-turbo-preview"
    rate_limit_requests_per_minute: 60
    rate_limit_tokens_per_minute: 100000
    models:
      - "gpt-4-turbo-preview"
      - "gpt-4"
      - "gpt-3.5-turbo"
      - "text-embedding-3-large"

  local:
    enabled: true
    base_url: "http://localhost:11434"
    default_model: "llama3.1:8b"
    timeout: 300
    models:
      - "llama3.1:8b"
      - "llama3.1:70b"
      - "mistral:7b"
      - "codellama:7b"

  anthropic:
    enabled: false
    default_model: "claude-3-sonnet-20240229"
    models:
      - "claude-3-sonnet-20240229"
      - "claude-3-haiku-20240307"

# Database Configuration
database:
  url: "sqlite:///./data/legal_agent_dev.db"
  echo: true
  pool_size: 5
  max_overflow: 10

# Security Configuration
security:
  token_expiry_hours: 24
  password_hash_rounds: 12
  enable_content_validation: true
  max_file_size_mb: 50
  allowed_file_extensions:
    - ".pdf"
    - ".docx"
    - ".txt"
    - ".md"
    - ".rtf"

# RAG Configuration
rag:
  vector_store_type: "chroma"
  embedding_model: "text-embedding-3-large"
  chunk_size: 1000
  chunk_overlap: 200
  max_retrieval_results: 10
  similarity_threshold: 0.7
  collection_name: "legal_documents_dev"

# Agent Configuration
agents:
  memory_limit: 150
  max_reasoning_steps: 50
  enable_collaboration: true
  default_temperature: 0.7
  max_tokens: 2000

# Dashboard Configuration
dashboard:
  host: "localhost"
  port: 8501
  debug: true
  theme: "light"
  enable_authentication: false

# API Configuration
api:
  host: "localhost"
  port: 8000
  debug: true
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8501"
  enable_docs: true

# Logging Configuration
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/legal_agent_dev.log"
  max_file_size_mb: 10
  backup_count: 5

# Data Directories
data_dir: "./data"
documents_dir: "./data/documents"
embeddings_dir: "./data/embeddings"
cache_dir: "./data/cache"
'''

# Production configuration
prod_config = '''# Production Configuration for Legal Agent Orchestrator

# LLM Provider Settings
llm_providers:
  openai:
    enabled: true
    default_model: "gpt-4-turbo-preview"
    rate_limit_requests_per_minute: 100
    rate_limit_tokens_per_minute: 200000
    models:
      - "gpt-4-turbo-preview"
      - "gpt-4"
      - "gpt-3.5-turbo"

  local:
    enabled: true
    base_url: "http://localhost:11434"
    default_model: "llama3.1:70b"
    timeout: 600
    models:
      - "llama3.1:70b"
      - "llama3.1:8b"
      - "mistral:7b"

  anthropic:
    enabled: true
    default_model: "claude-3-sonnet-20240229"
    rate_limit_requests_per_minute: 50
    models:
      - "claude-3-sonnet-20240229"
      - "claude-3-haiku-20240307"

# Database Configuration
database:
  url: "postgresql://user:password@localhost:5432/legal_agent_prod"
  echo: false
  pool_size: 20
  max_overflow: 30

# Security Configuration
security:
  token_expiry_hours: 8
  password_hash_rounds: 15
  enable_content_validation: true
  max_file_size_mb: 100
  allowed_file_extensions:
    - ".pdf"
    - ".docx"
    - ".txt"
    - ".md"

# RAG Configuration
rag:
  vector_store_type: "chroma"
  embedding_model: "text-embedding-3-large"
  chunk_size: 1000
  chunk_overlap: 200
  max_retrieval_results: 15
  similarity_threshold: 0.75
  collection_name: "legal_documents_prod"

# Agent Configuration
agents:
  memory_limit: 200
  max_reasoning_steps: 100
  enable_collaboration: true
  default_temperature: 0.5
  max_tokens: 4000

# Dashboard Configuration
dashboard:
  host: "0.0.0.0"
  port: 8501
  debug: false
  theme: "light"
  enable_authentication: true

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  cors_origins:
    - "https://yourdomain.com"
  enable_docs: false

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/var/log/legal_agent/legal_agent.log"
  max_file_size_mb: 50
  backup_count: 10

# Data Directories
data_dir: "/opt/legal_agent/data"
documents_dir: "/opt/legal_agent/data/documents"
embeddings_dir: "/opt/legal_agent/data/embeddings"
cache_dir: "/opt/legal_agent/data/cache"
'''

# Local configuration template
local_config = '''# Local Configuration for Legal Agent Orchestrator
# Copy this file to config/local.yaml and customize for your setup

# LLM Provider Settings
llm_providers:
  openai:
    enabled: true
    default_model: "gpt-4-turbo-preview"
    rate_limit_requests_per_minute: 30
    rate_limit_tokens_per_minute: 50000

  local:
    enabled: true
    base_url: "http://localhost:11434"
    default_model: "llama3.1:8b"
    timeout: 300

  anthropic:
    enabled: false

# Security Configuration
security:
  enable_content_validation: true
  max_file_size_mb: 25

# RAG Configuration
rag:
  vector_store_type: "chroma"
  embedding_model: "text-embedding-3-large"
  chunk_size: 800
  chunk_overlap: 150
  collection_name: "legal_documents_local"

# Dashboard Configuration
dashboard:
  host: "localhost"
  port: 8501
  debug: true
  enable_authentication: false

# API Configuration
api:
  host: "localhost"
  port: 8000
  debug: true
  enable_docs: true

# Logging Configuration
logging:
  level: "INFO"
  file: "./logs/legal_agent_local.log"
'''

# Environment template
env_template = '''# Environment Variables for Legal Agent Orchestrator
# Copy this file to .env and fill in your actual values

# Environment
ENVIRONMENT=development
DEBUG=true

# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Azure OpenAI (if using)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key_here

# Local LLM Configuration
OLLAMA_HOST=http://localhost:11434

# Database
DATABASE_URL=sqlite:///./data/legal_agent.db

# Security Keys (generate secure random values)
SECRET_KEY=your_secret_key_32_chars_minimum_here
ENCRYPTION_KEY=your_encryption_key_32_chars_minimum

# Data Directories
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents
EMBEDDINGS_DIR=./data/embeddings
CACHE_DIR=./data/cache

# Optional: Custom ports
DASHBOARD_PORT=8501
API_PORT=8000

# Optional: Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/legal_agent.log

# Optional: Rate Limiting
OPENAI_RATE_LIMIT_RPM=60
OPENAI_RATE_LIMIT_TPM=100000

# Optional: File Upload Limits
MAX_FILE_SIZE_MB=50

# Optional: Vector Database Settings
VECTOR_STORE_TYPE=chroma
EMBEDDING_MODEL=text-embedding-3-large
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
'''

# Write configuration files
with open("config/development.yaml", "w") as f:
    f.write(dev_config)

with open("config/production.yaml", "w") as f:
    f.write(prod_config)

with open("config/local.yaml", "w") as f:
    f.write(local_config)

with open(".env.template", "w") as f:
    f.write(env_template)

print("Configuration files created!")
print("- config/development.yaml")
print("- config/production.yaml") 
print("- config/local.yaml")
print("- .env.template")
# Create requirements.txt
requirements_content = """# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
streamlit==1.28.1
pydantic==2.5.0
pydantic-settings==2.0.3

# LLM and AI
openai==1.3.7
anthropic==0.3.11
langchain==0.0.339
langchain-community==0.0.20
langchain-openai==0.0.2
crewai==0.10.10
transformers==4.35.2
torch==2.1.1

# Local LLM support
ollama==0.1.7

# Vector database and embeddings
chromadb==0.4.18
sentence-transformers==2.2.2
faiss-cpu==1.7.4

# Document processing
pypdf2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6

# Data processing
pandas==2.1.3
numpy==1.24.3

# Database
sqlalchemy==2.0.23
alembic==1.12.1

# Security
cryptography==41.0.7
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
httpx==0.25.2

# Monitoring and logging
structlog==23.2.0
rich==13.7.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2

# Development
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# Configuration
python-dotenv==1.0.0
pyyaml==6.0.1

# Async support
asyncio==3.4.3
aiofiles==23.2.1
"""

with open("requirements.txt", "w") as f:
    f.write(requirements_content)

# Create pyproject.toml
pyproject_content = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "legal-agent-orchestrator"
version = "0.1.0"
description = "A personal agent orchestration app for lawyers with dynamic RAG and reasoning capabilities"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Legal AI Developer", email = "developer@legalai.com"}
]
keywords = ["legal", "ai", "agents", "rag", "llm", "orchestration"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Legal",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Office/Business :: Legal",
]
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "streamlit>=1.28.1",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.0.3",
    "openai>=1.3.7",
    "anthropic>=0.3.11",
    "langchain>=0.0.339",
    "langchain-community>=0.0.20",
    "langchain-openai>=0.0.2",
    "crewai>=0.10.10",
    "transformers>=4.35.2",
    "torch>=2.1.1",
    "ollama>=0.1.7",
    "chromadb>=0.4.18",
    "sentence-transformers>=2.2.2",
    "faiss-cpu>=1.7.4",
    "pypdf2>=3.0.1",
    "python-docx>=1.1.0",
    "python-multipart>=0.0.6",
    "pandas>=2.1.3",
    "numpy>=1.24.3",
    "sqlalchemy>=2.0.23",
    "alembic>=1.12.1",
    "cryptography>=41.0.7",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "httpx>=0.25.2",
    "structlog>=23.2.0",
    "rich>=13.7.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "aiofiles>=23.2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.7.1",
]

[project.urls]
"Homepage" = "https://github.com/your-username/legal-agent-orchestrator"
"Bug Tracker" = "https://github.com/your-username/legal-agent-orchestrator/issues"
"Repository" = "https://github.com/your-username/legal-agent-orchestrator"

[tool.setuptools.packages.find]
where = ["."]
include = ["legal_agent_orchestrator*"]

[tool.black]
line-length = 88
target-version = ['py39']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
"""

with open("pyproject.toml", "w") as f:
    f.write(pyproject_content)

print("Requirements and project configuration files created!")
import os
import json

# Create the main project structure
project_structure = {
    "legal_agent_orchestrator": {
        "__init__.py": "",
        "agents": {
            "__init__.py": "",
            "base_agent.py": "",
            "research_agent.py": "",
            "reasoning_agent.py": "",
            "drafting_agent.py": "",
            "review_agent.py": "",
            "orchestrator.py": ""
        },
        "llm_providers": {
            "__init__.py": "",
            "base_provider.py": "",
            "openai_provider.py": "",
            "azure_provider.py": "",
            "local_provider.py": "",
            "anthropic_provider.py": ""
        },
        "rag": {
            "__init__.py": "",
            "vector_store.py": "",
            "embeddings.py": "",
            "retriever.py": "",
            "document_processor.py": ""
        },
        "reasoning": {
            "__init__.py": "",
            "chain_of_thought.py": "",
            "legal_reasoning.py": "",
            "case_analysis.py": ""
        },
        "api": {
            "__init__.py": "",
            "routes.py": "",
            "models.py": "",
            "middleware.py": ""
        },
        "dashboard": {
            "__init__.py": "",
            "app.py": "",
            "components.py": "",
            "utils.py": ""
        },
        "utils": {
            "__init__.py": "",
            "config.py": "",
            "logging.py": "",
            "security.py": "",
            "validators.py": ""
        },
        "data": {
            "documents": {},
            "embeddings": {},
            "cache": {}
        }
    },
    "tests": {
        "__init__.py": "",
        "test_agents.py": "",
        "test_llm_providers.py": "",
        "test_rag.py": "",
        "test_reasoning.py": "",
        "test_api.py": ""
    },
    "docs": {
        "README.md": "",
        "API.md": "",
        "SETUP.md": "",
        "SECURITY.md": "",
        "COMPLIANCE.md": ""
    },
    "scripts": {
        "setup.py": "",
        "run_dev.py": "",
        "deploy.py": ""
    },
    "config": {
        "development.yaml": "",
        "production.yaml": "",
        "local.yaml": ""
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)

# Create the project structure
create_structure(".", project_structure)

print("Project structure created successfully!")
print("\nProject overview:")
for root, dirs, files in os.walk("legal_agent_orchestrator"):
    level = root.replace("legal_agent_orchestrator", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = " " * 2 * (level + 1)
    for file in files:
        print(f"{sub_indent}{file}")
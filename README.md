# Pharmacy RAG System

A **secure, enterprise-grade** Retrieval-Augmented Generation (RAG) system specifically designed for pharmacy and medical knowledge management. This system runs **completely locally** with no external API calls, ensuring **maximum data privacy and security** for healthcare organizations.

## Enterprise Security Features

- **100% Local Operation**: All models run locally via Ollama - no data leaves your infrastructure
- **No External API Calls**: Zero dependency on cloud services or third-party APIs
- **HIPAA-Ready Architecture**: Designed with healthcare data privacy in mind
- **Air-Gap Compatible**: Can operate in completely isolated network environments
- **Local Vector Storage**: ChromaDB runs locally with no external database connections

## Key Features

- **Pharmacy-Focused Knowledge Base**: Optimized for pharmaceutical, toxicology, and clinical information
- **Multi-Modal Document Support**: Process PDF, DOCX, and TXT medical documents
- **Advanced Retrieval**: Semantic search with configurable relevance scoring
- **Source Attribution**: Every answer includes specific document references and line numbers
- **Comprehensive Testing**: Industry-standard benchmarking for accuracy and safety
- **Performance Monitoring**: Built-in metrics for response time and quality assessment

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for optimal performance)
- **Storage**: 10GB+ free space for models and vector database
- **OS**: Linux, macOS, or Windows
- **Ollama**: Local LLM runtime environment

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
git clone https://github.com/Lapintam/pharmacy-rag-example.git
cd pharmacy-rag-example

# Run the automated setup script
python setup.py
```

The setup script will:
- Check system requirements
- Install Python dependencies
- Verify Ollama installation
- Check for required models
- Provide next steps guidance

### Option 2: Manual Setup

### 1. Clone and Setup

```bash
git clone https://github.com/Lapintam/pharmacy-rag-example.git
cd pharmacy-rag-example

# Create isolated virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama (Local LLM Runtime)

```bash
# Visit https://ollama.com for installation instructions
# Then pull the recommended model:
ollama pull llama3.2:3b

# Start Ollama service
ollama serve
```

### 3. Add Your Medical Documents

```bash
# Place your pharmacy/medical documents in the data directory
# Supported formats: PDF, DOCX, TXT
cp your_medical_documents/* data/
```

### 4. Build Knowledge Base

```bash
# Process and index all documents
python process_documents.py
```

### 5. Query the System

```bash
# Interactive querying
python query_data.py "What are the contraindications for beta-blockers?"

# With custom parameters
python query_data.py "Acetaminophen overdose treatment" --k 10 --model llama3.2:3b
```

## Quality Assurance & Testing

Run comprehensive benchmarks to validate system accuracy and safety:

```bash
# Run full test suite with industry-standard metrics
python test_rag_system.py
```

### Benchmark Categories

- **Accuracy Testing**: Keyword-based relevance scoring
- **Safety-Critical Validation**: Special focus on medication safety
- **Performance Metrics**: Response time and throughput analysis
- **Category-Specific Tests**: Cardiology, toxicology, pediatrics, etc.
- **Enterprise Readiness**: Reliability and consistency testing

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Medical       │    │   Document       │    │   Vector        │
│   Documents     │───▶│   Processor      │───▶│   Database      │
│   (PDF/DOCX)    │    │   (Local)        │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Query         │    │   RAG Engine     │◀────────────┘
│   Interface     │───▶│   (Local LLM)    │
│                 │    │   via Ollama     │
└─────────────────┘    └──────────────────┘
```

## Configuration Options

### Query Parameters

- `--k`: Number of relevant chunks to retrieve (default: 5)
- `--model`: Ollama model to use (default: "llama3.2:3b")
- `--format`: Output format - "text" or "json" (default: "text")

### Embedding Models

Modify `get_embedding_function.py` to use different embedding models:
- Default: `all-MiniLM-L6-v2` (fast, efficient)
- Alternative: `all-MiniLM-L12-v2` (higher accuracy)

### Document Processing

Adjust chunk size and overlap in `process_documents.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Adjust based on document complexity
    chunk_overlap=100,  # Increase for better context preservation
)
```

## Project Structure

```
pharmacy-rag-example/
├── data/                          # Medical document storage
│   ├── Cardiology/               # Cardiovascular medications
│   ├── Toxicology/               # Poison control & antidotes
│   ├── Pediatrics/               # Pediatric dosing guidelines
│   ├── Infectious Disease/       # Antimicrobial therapy
│   └── ...                       # Additional medical specialties
├── chroma/                       # Vector database (auto-generated)
├── process_documents.py          # Document ingestion pipeline
├── query_data.py                 # Query interface and RAG engine
├── get_embedding_function.py     # Embedding model configuration
├── test_rag_system.py           # Comprehensive testing suite
├── setup.py                      # Automated setup and system check
└── requirements.txt              # Python dependencies
```

## Medical Specialties Supported

The system is pre-configured for these pharmacy/medical domains:

- **Cardiology**: Heart medications, contraindications
- **Toxicology**: Antidotes, poison management
- **Pediatrics**: Weight-based dosing, age considerations
- **Infectious Disease**: Antimicrobial selection
- **Emergency Medicine**: Critical care protocols
- **Endocrinology**: Diabetes management, hormones
- **Neurology**: CNS medications, seizure management
- **OBGYN**: Pregnancy safety, contraceptives

## Security & Compliance

### Data Privacy
- **No Cloud Dependencies**: All processing occurs locally
- **Encrypted Storage**: Vector database can be encrypted at rest
- **Access Control**: Implement file-system level permissions
- **Audit Logging**: Track all queries and system access

### Compliance Considerations
- **HIPAA**: Architecture supports HIPAA compliance requirements
- **SOC 2**: Local operation eliminates many third-party risks
- **GDPR**: No data transmission to external services
- **FDA 21 CFR Part 11**: Audit trail capabilities for regulated environments

## Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for optimal model performance
- **Storage**: SSD for faster vector database operations
- **GPU**: Optional, but improves embedding generation speed

### Scaling Considerations
- **Horizontal Scaling**: Deploy multiple instances behind load balancer
- **Database Optimization**: Tune ChromaDB settings for large document sets
- **Model Selection**: Balance accuracy vs. speed based on use case

## Troubleshooting

### Common Issues

**Database Not Found**
```bash
# Rebuild the vector database
python process_documents.py
```

**Ollama Connection Issues**
```bash
# Ensure Ollama is running
ollama serve

# Verify model availability
ollama list
```

**Poor Query Results**
```bash
# Run benchmarks to identify issues
python test_rag_system.py

# Check document quality and relevance
```

**Memory Issues**
- Reduce chunk size in `process_documents.py`
- Use smaller embedding model
- Limit concurrent queries

## Additional Resources

- [Ollama Documentation](https://ollama.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Healthcare AI Best Practices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all benchmarks pass
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Medical Disclaimer

This system is designed for informational purposes and clinical decision support. It should not replace professional medical judgment or be used as the sole basis for patient care decisions. Always consult current medical literature and follow institutional protocols.

# DocIntel - Document Intelligence System

A comprehensive document intelligence system that performs data preparation, information extraction, text summarization, and agentic reasoning.

## 🎯 Overview

DocIntel is designed to extract insights from large document collections using:
- **Data Preparation**: Text preprocessing and exploratory analysis
- **Information Extraction**: Entity recognition using multiple approaches
- **Text Summarization**: Both extractive and abstractive methods
- **Agentic Design**: Research agent that chains operations for complex queries

## 🏗️ Architecture

```
DocIntel/
├── src/
│   ├── data_loader.py      # Dataset loading and management
│   ├── preprocessing.py    # Text preprocessing pipeline
│   ├── extractor.py        # Entity extraction methods
│   ├── summarizer.py       # Text summarization techniques
│   └── agent.py            # Agentic reasoning system
├── notebooks/
│   ├── exploration.ipynb   # Data exploration and analysis
│   └── evaluation.ipynb    # Model evaluation and results
├── data/                   # Dataset storage
├── results/                # Output files and reports
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Run Data Exploration**
```bash
python src/data_loader.py
```

3. **Process Documents**
```bash
python src/preprocessing.py
python src/extractor.py
python src/summarizer.py
```

4. **Run Agent**
```bash
python src/agent.py
```

## 📊 Features

### Phase 1: Data Preparation
- Multiple dataset support (Reuters, Kaggle datasets, ArXiv)
- Comprehensive text preprocessing
- Exploratory data analysis with visualizations

### Phase 2: Information Extraction
- Multi-approach entity extraction (Regex, SpaCy, Transformers)
- Extractive summarization (TextRank, TF-IDF)
- Optional abstractive summarization (T5, BART)

### Phase 3: Agentic Design
- Goal-oriented research agent
- Tool-based architecture
- Chain-of-thought reasoning

## 🎯 Agent Capabilities

The research agent can:
- Analyze document collections
- Extract relevant entities and insights
- Generate comprehensive summaries
- Answer complex queries by chaining operations

Example query: *"Summarize the top issues customers are facing from support tickets over the last 7 days."*

## 🔧 Configuration

Edit configuration parameters in each module or create a `config.py` file for centralized settings.

## 📈 Evaluation

The system includes evaluation metrics:
- Manual inspection tools
- ROUGE scores for summarization
- Entity extraction accuracy
- Agent response quality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

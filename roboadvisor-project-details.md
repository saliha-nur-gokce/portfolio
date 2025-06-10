# AI-Powered Financial Assistant: Fine-Tuned and Retrieval-Based LLM for Corporate QA

This project presents a modular AI system for corporate financial question answering (QA), developed as part of a Generative AI and Deep Learning course. The system is designed to operate on structured Turkish financial data and evaluates two distinct model architectures: instruction-based fine-tuning of LLaMA 3 and an enhanced Retrieval-Augmented Generation (RAG) pipeline with executable reasoning.

## Architecture and Model Design

The project started with a standard **RAG** setup, where relevant financial passages were retrieved using semantic search (FAISS + SentenceTransformers) and passed to a language model to generate answers. However, we identified performance limitations in numerical precision and reasoning over structured data. To overcome these, we transitioned to an improved system called **ragenh**.

- **Enhanced RAG (ragenh)**: This version integrates `pandas` DataFrames directly into the generation loop. Rather than treating financial text as static, the model generates executable Python code to query structured tables — enabling accurate responses for numeric lookups, temporal comparisons, and investment suggestions.

- **Fine-tuning**: In parallel, we trained a LLaMA 3 8B model using LoRA adapters and Turkish instruction-based financial prompts. The fine-tuned model specialized in understanding financial question structure and producing coherent, domain-aligned outputs without relying on external retrieval.

Each financial domain — such as **liquidity**, **investment planning**, and **debt strategy** — is implemented as a modular QA task, enabling scalable development and focused evaluation. Datasets include balance sheets and income statements from BIST100 companies (2008–2024), collected via Turkey’s Public Disclosure Platform (KAP) and enriched with derived financial ratios and trends.

## Evaluation and Metrics

To assess model performance, we developed a custom evaluation pipeline covering three question types: factual lookup, trend analysis, and opinion-based investment reasoning. Evaluation metrics include:

- **ROUGE-L** for lexical similarity  
- **BERTScore (F1)** for semantic alignment using Turkish embeddings  
- **Number Match Score** for numeric accuracy  
- **Trend Direction Agreement** for temporal consistency  

### Key Findings

- **Fine-tuned model** produced semantically rich and fluent answers, particularly in interpretive and trend questions.  
- **RAG** performed best in raw data extraction, but lacked deeper contextualization.  
- **ragenh** achieved the best balance — combining accuracy with grounded, structured reasoning, especially in open-ended and strategic queries.

Evaluation demonstrated that `ragenh` outperformed the other models in complex reasoning tasks thanks to its ability to dynamically interact with real tabular data.

![Evaluation Metrics](images/evaluation.png)

## Tools and Technologies

- Python  
- `transformers`, `peft` – for instruction tuning (LoRA adapters)  
- `sentence-transformers`, `faiss` – for semantic retrieval  
- `pandas`, `numpy`, `matplotlib` – for data processing and visualization  
- Custom evaluation scripts for financial metrics and Turkish-language normalization  

## Impact and Future Work

This project contributes to the development of **interpretable, low-resource financial AI tools** tailored for the Turkish language and corporate finance domain. By evolving from a standard RAG model to a reasoning-enhanced architecture, it highlights the importance of structured data access and modular QA design.

Planned extensions include:

- Integration of macroeconomic and unstructured data (e.g., analyst reports, news)
- Full adaptation of the system to serve Turkish-speaking SMEs as a financial assistant
- Improving explainability layers for end-user transparency
- Experimentation with more complex reasoning and forecasting modules

---

[View Project Paper](https://drive.google.com/file/d/164D1TA2y1MSHfCvgd1iv4KgvSRxLuu5x/view?usp=sharing)  
[View Project Codes](link_to_repo_or_folder)

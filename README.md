# DOCBOT: Offline Document Assistant

## Introduction

DOCBOT is an offline document assistant developed by Mesa Sarah Vasantha Zephyr and Rampalli Prajna Paramita as an internship project report at the Department of Advanced Systems Laboratory, Defence Research and Development Organization, Hyderabad in November 2023. It is designed to provide intelligent document assistance without the need for an internet connection.

## Abstract

DOCBOT utilizes machine learning and Langchain technology to extract and summarize information from uploaded PDF documents. Its offline capabilities make it invaluable for individuals and organizations operating in environments with limited or no internet access.

## Table of Contents

1. Introduction
   - Problem Definition
   - Solution Approach
   - Technologies Used
2. Background
   - Artificial Intelligence
   - Machine Learning
   - Large Language Models (LLMs)
3. System Architecture
   - Langchain
4. Implementation
5. Testing
6. Deployment
7. Maintenance
8. Results
9. Conclusion
10. Future Work
11. References

## Implementation Overview

- **User Interface**: Implemented using Streamlit, allowing users to upload PDF documents and interact with the system.
- **PDF Text Extraction**: Utilizes PyPDF2 library to extract text from uploaded PDFs.
- **Text Processing**: Processes extracted text into a format suitable for question answering using Langchain and Chroma for sentence embeddings and indexing.
- **Question Answering System**: Core component that retrieves relevant documents and generates answers using Langchain.

## Testing

- **Unit Testing**: Tested individual modules such as File Uploader, PDF Reader, Text Processor, and Question Answering System.
- **Integration Testing**: Ensured smooth interaction between different system components.
- **System Testing**: Evaluated the system's ability to answer questions accurately and efficiently.

## Deployment

DOCBOT can be deployed on Streamlit Community Cloud with just one click. It takes care of server management, scaling, and infrastructure, making deployment hassle-free.

## Maintenance

Streamlit Community Cloud provides tools for effective app management, including log viewing, version history, activity monitoring, environment variables management, and startup commands.

## Results

DOCBOT achieved an 85% accuracy rate in answering user queries, demonstrating its effectiveness in retrieving information from documents.

## Conclusion

DOCBOT represents a significant advancement in offline document assistance, offering accessibility, privacy, and versatility. Its integration of machine learning and Langchain technologies ensures continuous improvement and adaptability.

## Future Work

Potential enhancements for DOCBOT include expanding document support, enhancing question answering capabilities, integrating knowledge graphs, and enabling multimodal interaction.

## References

- Deep Learning by kenovy
- Quick overview: what is AI (Artificial Intelligence), ML (Machine Learning), and DL (Deep Learning) by Jalel TOUNSI
- Streamlit Documentation
- Hugging Face Documentation


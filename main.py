import os
import shutil
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import logging
from logging.handlers import RotatingFileHandler
from llama_parse import LlamaParse
from openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
import traceback
import sys
from functools import lru_cache
import numpy as np
# testgit
# Load environment variables
load_dotenv()

# Network connectivity test
def test_openai_connectivity():
    """测试 OpenAI API 连接"""
    try:
        import requests
        response = requests.get('https://dashscope.aliyuncs.com/compatible-mode/v1', timeout=10)
        return True, f"连接正常，状态码: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "连接超时 - 可能需要代理或网络有问题"
    except requests.exceptions.ConnectionError as e:
        return False, f"连接错误: {str(e)}"
    except Exception as e:
        return False, f"未知错误: {str(e)}"

# Test connectivity at startup
connectivity_ok, connectivity_msg = test_openai_connectivity()
print(f"OpenAI API 连接测试: {connectivity_msg}")
if not connectivity_ok:
    print("警告: OpenAI API 连接失败，embeddings 功能可能无法正常工作")
    print("请检查网络连接、代理设置或防火墙配置")

app = Flask(__name__)

# Custom formatter for logging
class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.request_id = getattr(record, 'request_id', '-')
        return super().format(record)

# Logging configuration
log_formatter = CustomFormatter('%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s')
log_file = 'app.log'
log_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
log_handler.setFormatter(log_formatter)
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
app.logger.addHandler(console_handler)

# Constants
TOP_K = 6
OUTPUT_FOLDER = 'parsed_pdfs'
FAISS_INDEX_FOLDER = 'faiss_index'
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary directories
for folder in [OUTPUT_FOLDER, FAISS_INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Environment variable handling
def get_required_env_var(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        app.logger.critical(f"{var_name} not found in environment variables")
        raise ValueError(f"{var_name} not found in environment variables")
    return value

# Initialize clients
openai_api_key = get_required_env_var("OPENAI_API_KEY")
llama_cloud_api_key = get_required_env_var("LLAMA_CLOUD_API_KEY")

client = OpenAI(
    api_key=openai_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )

# Initialize LlamaParse
parser = LlamaParse(
    api_key=llama_cloud_api_key,
    api_result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
    num_workers=4,
    verbose=True,
    language="en"
)

# Initialize ChatOpenAI with caching
@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="qwen-plus",  # 使用qwen3-32b的非思考版本
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        streaming=False,  # 非流式输出
    )

# Pydantic models
class GapItem(BaseModel):
    description: str = Field(description="Description of the gap between RFP and Response")
    severity: Literal["Low", "Medium", "High"] = Field(description="Severity of the gap")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "Missing technical specifications",
                    "severity": "High"
                }
            ]
        }
    }

class GapAnalysis(BaseModel):
    summary: str = Field(description="Brief summary of the overall gap analysis")
    gaps: List[GapItem] = Field(description="List of identified gaps")
    suggestions: List[str] = Field(description="List of suggestions to address the gaps")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Several gaps identified between RFP and Response",
                    "gaps": [
                        {
                            "description": "Missing technical specifications",
                            "severity": "High"
                        }
                    ],
                    "suggestions": [
                        "Include detailed technical specifications"
                    ]
                }
            ]
        }
    }

# Custom exceptions
class DocumentProcessingError(Exception):
    pass

class RetrieverError(Exception):
    pass

# FAISS operations
class FAISSOperations:
    @staticmethod
    def clear_index(collection_name: str) -> None:
        try:
            index_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                app.logger.info(f"Cleared FAISS index for collection: {collection_name}")
        except Exception as e:
            app.logger.error(f"Error clearing FAISS index for collection {collection_name}: {str(e)}")
            app.logger.debug(traceback.format_exc())
            raise DocumentProcessingError(f"Failed to clear FAISS index: {str(e)}")

    @staticmethod
    def create_index(documents: List[Document], collection_name: str) -> FAISS:
        try:
            # 使用自定义的 DashScope embeddings
            embeddings = DashScopeEmbeddings(
                api_key=openai_api_key,
                model="text-embedding-v3"
            )
            
            # 过滤和清理文档
            valid_documents = []
            for doc in documents:
                if doc.page_content and doc.page_content.strip():
                    # 确保文本是字符串且不为空
                    cleaned_text = str(doc.page_content).strip()
                    if len(cleaned_text) > 0:
                        valid_documents.append(Document(
                            page_content=cleaned_text,
                            metadata=doc.metadata
                        ))
            
            if not valid_documents:
                raise DocumentProcessingError("没有有效的文档内容可以处理")
            
            texts = [doc.page_content for doc in valid_documents]
            metadatas = [doc.metadata for doc in valid_documents]
            
            # 添加调试信息
            app.logger.info(f"开始创建 FAISS 索引，有效文档数量: {len(valid_documents)} (原始: {len(documents)})")
            if texts:
                app.logger.info(f"文本长度统计: 最小={min(len(t) for t in texts)}, 最大={max(len(t) for t in texts)}, 平均={sum(len(t) for t in texts)//len(texts)}")
            
            # 手动创建向量索引，避免 LangChain 兼容性问题
            app.logger.info("使用自定义 DashScope embeddings 创建向量...")
            
            # 分批处理以避免超时
            batch_size = 5  # 增加批次大小
            all_embeddings = []
            processed_texts = []
            processed_metadatas = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                # 再次验证批次中的文本
                clean_batch_texts = [str(text).strip() for text in batch_texts if text and str(text).strip()]
                if not clean_batch_texts:
                    app.logger.warning(f"批次 {i//batch_size + 1} 没有有效文本，跳过")
                    continue
                
                app.logger.info(f"处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}，文本数量: {len(clean_batch_texts)}")
                
                try:
                    # 获取嵌入向量
                    batch_embeddings = embeddings.embed_documents(clean_batch_texts)
                    
                    # 记录成功处理的文本和向量
                    all_embeddings.extend(batch_embeddings)
                    processed_texts.extend(clean_batch_texts)
                    processed_metadatas.extend(batch_metadatas[:len(clean_batch_texts)])
                    
                    app.logger.info(f"批次 {i//batch_size + 1} 处理成功，获得 {len(batch_embeddings)} 个向量")
                    
                except Exception as batch_error:
                    app.logger.error(f"批次 {i//batch_size + 1} 处理失败: {str(batch_error)}")
                    # 跳过这个批次，继续处理下一个
                    continue
            
            if not all_embeddings:
                raise DocumentProcessingError("所有批次都失败了，无法创建向量索引")
            
            app.logger.info(f"成功处理 {len(all_embeddings)} 个文档，开始创建 FAISS 索引...")
            
            # 创建 FAISS 索引
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # 使用 LangChain 的 OpenAIEmbeddings 作为包装器，但仅用于 FAISS 索引创建
            dummy_embeddings = OpenAIEmbeddings(
                model="text-embedding-v3",
                api_key=openai_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            # 创建 FAISS 索引时使用预计算的向量
            vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(processed_texts, all_embeddings)),
                embedding=dummy_embeddings,
                metadatas=processed_metadatas
            )

            index_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)
            os.makedirs(index_path, exist_ok=True)
            vectorstore.save_local(index_path)

            app.logger.info(f"Created new FAISS index for collection: {collection_name}")
            return vectorstore
        except Exception as e:
            app.logger.error(f"Error creating FAISS index for collection {collection_name}: {str(e)}")
            app.logger.debug(traceback.format_exc())
            raise DocumentProcessingError(f"Failed to create FAISS index: {str(e)}")

# Document Processing
class DocumentProcessor:
    @staticmethod
    def parse_pdf(file_path: str, output_name: str) -> str:
        try:
            FAISSOperations.clear_index(output_name)

            result = parser.load_data(file_path)

            output_path = os.path.join(OUTPUT_FOLDER, f"{output_name}.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                for page in result:
                    f.write(page.text)
                    f.write("\n\n---\n\n")

            documents = [
                Document(page_content=page.text, metadata={"source": output_name, "page": i})
                for i, page in enumerate(result)
            ]
            # print(111)
            FAISSOperations.create_index(documents, output_name)
            return f"Successfully processed {output_name}"

        except Exception as e:
            app.logger.error(f"Error parsing {file_path}: {str(e)}")
            app.logger.debug(traceback.format_exc())
            raise DocumentProcessingError(f"Failed to parse PDF: {str(e)}")

# Document Retrieval
class DocumentRetriever:
    @staticmethod
    def initialize_retriever(collection_name: str) -> Optional[Any]:
        try:
            # 检查网络连接
            if not connectivity_ok:
                app.logger.warning(f"网络连接不可用，尝试加载本地 FAISS 索引: {collection_name}")
                
            # 使用自定义的 DashScope embeddings
            embeddings = DashScopeEmbeddings(
                api_key=openai_api_key,
                model="text-embedding-v3"
            )
            
            # 为了兼容性，还需要一个 LangChain embeddings 包装器
            dummy_embeddings = OpenAIEmbeddings(
                model="text-embedding-v3",
                api_key=openai_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            index_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)

            if not os.path.exists(index_path):
                app.logger.warning(f"No FAISS index found for collection: {collection_name}")
                return None

            vectorstore = FAISS.load_local(index_path, dummy_embeddings, allow_dangerous_deserialization=True)
            
            # 创建一个自定义的检索器，使用我们的 DashScope embeddings 进行查询
            class CustomRetriever:
                def __init__(self, vectorstore, custom_embeddings):
                    self.vectorstore = vectorstore
                    self.custom_embeddings = custom_embeddings
                
                def invoke(self, query: str) -> List[Document]:
                    # 使用自定义 embeddings 进行查询向量化
                    query_embedding = self.custom_embeddings.embed_query(query)
                    
                    # 直接在 FAISS 索引中搜索
                    docs_and_scores = self.vectorstore.similarity_search_by_vector(
                        query_embedding, k=TOP_K
                    )
                    return docs_and_scores
            
            return CustomRetriever(vectorstore, embeddings)
            
        except Exception as e:
            app.logger.error(f"Error initializing retriever for {collection_name}: {str(e)}")
            app.logger.debug(traceback.format_exc())
            raise RetrieverError(f"Failed to initialize retriever: {str(e)}")

    @staticmethod
    def retrieve_documents(query: str, retriever: Any) -> str:
        try:
            docs = retriever.invoke(query)
            return "\n\n".join([
                f"**Document {i+1}:**\n{doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
        except Exception as e:
            app.logger.error(f"Error retrieving documents: {str(e)}")
            raise RetrieverError(f"Failed to retrieve documents: {str(e)}")

# Analysis
class Analyzer:
    @staticmethod
    def analyze_gap(context: str) -> str:
        try:
            llm = get_llm()
            gap_analysis_parser = PydanticOutputParser(pydantic_object=GapAnalysis)

            gap_analysis_prompt = PromptTemplate(
                template="Analyze the gap between the RFP requirements and the Response based on the following context:\n\n{context}\n\n{format_instructions}\n",
                input_variables=["context"],
                partial_variables={"format_instructions": gap_analysis_parser.get_format_instructions()},
            )

            output = llm.invoke(gap_analysis_prompt.format(context=context))
            parsed_output = gap_analysis_parser.parse(output.content)

            result = parsed_output.model_dump()

            return f"""
Summary: {result['summary']}

Gaps:
{chr(10).join([f"- {gap['description']} (Severity: {gap['severity']})" for gap in result['gaps']])}

Suggestions:
{chr(10).join([f"- {suggestion}" for suggestion in result['suggestions']])}
"""
        except Exception as e:
            app.logger.error(f"Error during gap analysis: {str(e)}")
            raise ValueError(f"Failed to analyze gap: {str(e)}")

    @staticmethod
    def generate_insights(context: str) -> str:
        try:
            llm = get_llm()
            insight_prompt = f"""
            Based on the following documents:

            {context}

            Please provide a structured report with the following sections:

            1. Executive Summary:
               - Provide a concise overview of the main points from both the RFP and the Response.

            2. RFP Requirements Checklist:
               - List the critical requirements from the RFP.
               - For each requirement, indicate whether it is addressed in the Response (Addressed/Partially Addressed/Not Addressed).

            3. Key Insights:
               - Bullet point the most critical insights derived from comparing the RFP and the Response.
               - For each insight, provide a brief explanation of its significance.

            4. Trends and Patterns:
               - Identify and explain any common themes or patterns across both documents.

            5. Comparative Analysis:
               - Highlight notable differences between the RFP requirements and the Response.
               - Identify any areas where the Response exceeds RFP expectations.
            """

            insights = llm.invoke(insight_prompt)
            return insights.content
        except Exception as e:
            app.logger.error(f"Error generating insights: {str(e)}")
            raise ValueError(f"Failed to generate insights: {str(e)}")

# Report Formatting
class ReportFormatter:
    @staticmethod
    def format_report(raw_data: str) -> str:
        try:
            prompt = f"""
            Format the following raw data into a well-structured HTML report:

            {raw_data}

            """

            response = client.chat.completions.create(
                model="qwen-plus",  # 使用qwen3-32b非思考版本
                messages=[
                    {
                        "role": "system", 
                        "content": "You will be given unformated data and you will format it into a well-structured HTML report. You can use tables and images to make the report more readable."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                stream=False,  # 非流式输出
                response_format={"type": "text"}
            )

            return response.choices[0].message.content
        except Exception as e:
            app.logger.error(f"Error formatting report: {str(e)}")
            return f"<p>Error formatting report: {str(e)}</p>"

# Agent Tools
class AgentTools:
    @staticmethod
    @tool
    def retrieve_rfp_documents(query: str) -> str:
        """Retrieve relevant RFP documents using the query."""
        try:
            retriever = DocumentRetriever.initialize_retriever("rfp_parsed")
            if not retriever:
                return "Error: RFP documents not processed yet."

            return DocumentRetriever.retrieve_documents(query, retriever)
        except Exception as e:
            app.logger.error(f"Error retrieving RFP documents: {str(e)}")
            return f"Error retrieving RFP documents: {str(e)}"

    @staticmethod
    @tool
    def retrieve_response_documents(query: str) -> str:
        """Retrieve relevant Response documents using the query."""
        try:
            retriever = DocumentRetriever.initialize_retriever("response_parsed")
            if not retriever:
                return "Error: Response documents not processed yet."

            return DocumentRetriever.retrieve_documents(query, retriever)
        except Exception as e:
            app.logger.error(f"Error retrieving Response documents: {str(e)}")
            return f"Error retrieving Response documents: {str(e)}"

    @staticmethod
    def setup_agent():
        try:
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )

            tools = [
                Tool(
                    name="Retrieve RFP Documents",
                    func=AgentTools.retrieve_rfp_documents,
                    description="Retrieve relevant RFP documents using the query."
                ),
                Tool(
                    name="Retrieve Response Documents",
                    func=AgentTools.retrieve_response_documents,
                    description="Retrieve relevant Response documents using the query."
                ),
                Tool(
                    name="Analyze Gap",
                    func=Analyzer.analyze_gap,
                    description="Analyze gaps between RFP requirements and Response."
                ),
                Tool(
                    name="Generate Insights",
                    func=Analyzer.generate_insights,
                    description="Generate detailed insights from documents."
                )
            ]

            return initialize_agent(
                tools,
                get_llm(),
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        except Exception as e:
            app.logger.error(f"Error setting up agent: {str(e)}")
            raise ValueError(f"Failed to setup agent: {str(e)}")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_documents():
    if 'rfp' not in request.files or 'response' not in request.files:
        app.logger.warning("Incomplete request: Both RFP and Response files are required")
        return jsonify({"error": "Both RFP and Response files are required"}), 400

    rfp_file = request.files['rfp']
    response_file = request.files['response']

    # Validate file extensions
    for file in [rfp_file, response_file]:
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

    # Save uploaded files temporarily
    rfp_path = "temp_rfp.pdf"
    response_path = "temp_response.pdf"

    try:
        rfp_file.save(rfp_path)
        response_file.save(response_path)

        processor = DocumentProcessor()
        rfp_result = processor.parse_pdf(rfp_path, "rfp_parsed")
        response_result = processor.parse_pdf(response_path, "response_parsed")

        return jsonify({
            "rfp_result": rfp_result,
            "response_result": response_result,
            "message": "Documents processed successfully"
        })
    except Exception as e:
        app.logger.error(f"Error processing documents: {str(e)}")
        app.logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files
        for path in [rfp_path, response_path]:
            if os.path.exists(path):
                os.remove(path)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        retriever = DocumentRetriever()
        rfp_retriever = retriever.initialize_retriever("rfp_parsed")
        response_retriever = retriever.initialize_retriever("response_parsed")

        if not rfp_retriever or not response_retriever:
            raise ValueError("Documents must be processed before generating a report")

        rfp_content = retriever.retrieve_documents("Retrieve all relevant RFP content.", rfp_retriever)
        response_content = retriever.retrieve_documents("Retrieve all relevant Response content.", response_retriever)

        analyzer = Analyzer()
        raw_analysis = analyzer.analyze_gap(f"RFP Content:\n{rfp_content}\n\nResponse Content:\n{response_content}")
        raw_insights = analyzer.generate_insights(f"RFP Content:\n{rfp_content}\n\nResponse Content:\n{response_content}")

        raw_report = f"""
        # RFP and Response Analysis Report

        ## Part 1: Gap Analysis
        {raw_analysis}

        ## Part 2: Detailed Insights
        {raw_insights}
        """

        formatter = ReportFormatter()
        formatted_report = formatter.format_report(raw_report)
        return jsonify({"structured_report": formatted_report})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error generating report: {str(e)}")
        app.logger.debug(traceback.format_exc())
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        query = request.json.get('query')
        if not query:
            raise ValueError("No query provided")

        agent = AgentTools.setup_agent()
        if not agent:
            raise ValueError("Failed to initialize the agent")

        # Retrieve relevant documents
        rfp_docs = AgentTools.retrieve_rfp_documents(query)
        response_docs = AgentTools.retrieve_response_documents(query)

        enhanced_query = f"""
        Considering the following document contents:

        RFP Documents:
        {rfp_docs}

        Response Documents:
        {response_docs}

        Please answer the following query:
        {query}
        """

        result = agent.run(input=enhanced_query)
        app.logger.info(f"Chat query processed successfully: {query[:50]}...")
        return jsonify({"response": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error during chat execution: {str(e)}")
        app.logger.debug(traceback.format_exc())
        return jsonify({"error": "An internal error occurred"}), 500

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "An internal server error occurred"}), 500

# Custom DashScope Embeddings class
class DashScopeEmbeddings:
    """自定义的 DashScope embeddings 类，解决 LangChain 兼容性问题"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-v3", base_url: str = None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        try:
            # 确保输入是字符串列表
            clean_texts = []
            for text in texts:
                if isinstance(text, str) and text.strip():
                    clean_texts.append(text.strip())
            
            if not clean_texts:
                raise ValueError("没有有效的文本内容")
            
            # 调用 DashScope API
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_texts  # 直接传递字符串列表
            )
            
            return [data.embedding for data in response.data]
        except Exception as e:
            app.logger.error(f"DashScope embeddings 失败: {str(e)}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("查询文本不能为空")
        
        result = self.embed_documents([text.strip()])
        return result[0] if result else []

if __name__ == "__main__":
    try:
        # Ensure all required directories exist
        for directory in [OUTPUT_FOLDER, FAISS_INDEX_FOLDER]:
            os.makedirs(directory, exist_ok=True)

        # Validate environment variables
        required_vars = ["OPENAI_API_KEY", "LLAMA_CLOUD_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Start the Flask application
        port = int(os.environ.get('PORT', 5001))
        app.run(
            debug=False, 
            host='0.0.0.0', 
            port=port,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        app.logger.critical(f"Failed to start application: {str(e)}")
        app.logger.debug(traceback.format_exc())
        sys.exit(1)

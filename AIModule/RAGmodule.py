import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ConfigModule.ConfigManager import config


class RAGProcessor:
    def __init__(self):
        # 初始化配置
        self.novel_data_dir = Path("ReferDatabase/Novels")
        self.index_dir = Path("ReferDatabase/RAG_index")
        
        self.embed_model = OllamaEmbedding(
            model_name=config.get("embedding_model", "bge-m3:latest"),
            base_url="http://localhost:11434",
        )

        self.llm = Ollama(
            base_url="http://localhost:11434",
            model=config.get("llm_model"),
            temperature=config.get("temperature", 0.7),
            request_timeout=300
        )
        
        # 分块解析器配置
        self.node_parser = SentenceWindowNodeParser(
            window_size=512,
            window_metadata_key="context_window",
            original_text_metadata_key="original_content",
        )
        
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = config.get("chunk_size", 512)
        Settings.chunk_overlap = 128
    
    def _process_large_txt(self, file_path: Path) -> List[Document]:
        """流式处理大型TXT文件"""
        documents = []
        chunk_id = 0
        buffer = []
        buffer_size = 0  # 按字节计算
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                buffer.append(line)
                buffer_size += len(line.encode('utf-8'))
                
                # 每积累500KB处理一次
                if buffer_size >= 512 * 1024:
                    text_chunk = "".join(buffer)
                    documents.append(Document(
                        text=text_chunk,
                        metadata={
                            "source": str(file_path),
                            "chunk_id": chunk_id,
                            "total_chunks": "processing",
                            "type": "content",
                            "create_time": datetime.now().isoformat()
                        }
                    ))
                    chunk_id += 1
                    buffer = []
                    buffer_size = 0
            
            # 处理剩余内容
            if buffer:
                text_chunk = "".join(buffer)
                documents.append(Document(
                    text=text_chunk,
                    metadata={
                        "source": str(file_path),
                        "chunk_id": chunk_id,
                        "total_chunks": chunk_id + 1,
                        "type": "content",
                        "create_time": datetime.now().isoformat()
                    }
                ))
        return documents

    def _load_documents(self) -> List[Document]:
        """加载所有文档数据"""
        documents = []
        
        # 加载TXT文件
        for txt_file in self.novel_data_dir.glob("*.txt"):
            if txt_file.stat().st_size > 50 * 1024 * 1024:  # 50MB以上大文件
                documents.extend(self._process_large_txt(txt_file))
            else:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    documents.append(Document(
                        text=f.read(),
                        metadata={
                            "source": str(txt_file),
                            "type": "content",
                            "create_time": datetime.now().isoformat()
                        }
                    ))
        
        return documents

    def build_index(self, force_rebuild: bool = False) -> None:
        """构建/更新索引"""
        if not force_rebuild and (self.index_dir / "docstore.json").exists():
            print("🔄 检测到已有索引，使用缓存")
            return
            
        print("⏳ 开始构建索引...")
        documents = self._load_documents()
        if not documents:
            raise ValueError("未找到可索引的数据文件")
        
        # 创建索引
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=StorageContext.from_defaults(),
            show_progress=True
        )
        
        # 持久化存储
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=self.index_dir)
        print(f"✅ 索引构建完成，保存至 {self.index_dir}")

    def get_query_engine(self, similarity_top_k: int = 5):
        """获取查询引擎"""
        if not (self.index_dir / "docstore.json").exists():
            self.build_index()
            
        storage_context = StorageContext.from_defaults(
            persist_dir=self.index_dir
        )
        
        index = load_index_from_storage(
            storage_context=storage_context,
        )
        
        return index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="tree_summarize"
        )

    def GenerateWithOllama(self, prompt: str, top_k: int = 5) -> str:
        print("RAG GenerateWithOllama")
        """执行RAG查询"""
        try:
            query_engine = self.get_query_engine(top_k)
            response = query_engine.query(prompt)
            return str(response).strip()
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"
        

# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = RAGProcessor()
    
    # 构建索引（首次运行会较慢）
    processor.build_index()
    
    # 示例查询
    test_queries = [
        "小说的主角有哪些特殊能力？",
        "故事的主要冲突是什么？",
        "列出所有修仙相关的设定"
    ]
    
    common_prefix = "请以中文回答，并尽量详细。"

    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("📖 响应:", processor.GenerateWithOllama(common_prefix + query))
        print("━" * 50)


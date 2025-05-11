import sys
from pathlib import Path
# 自动计算项目根目录路径（根据实际文件层级调整）
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import faiss
from functools import lru_cache
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ConfigModule.ConfigManager import config


class RAGProcessor:
    def __init__(self):
        # 初始化配置
        self.novel_data_dir = Path("ReferDatabase/Novels")
        self.index_dir = Path("ReferDatabase/RAG_index")
        self.index = None

        self._init_models()

        if self._index_exists():
            print("检测到已有索引，尝试加载...")
            self._load_existing_index()
        else:
            print("未检测到索引，初始化新存储")
            self._build_index()
        self.index = load_index_from_storage(self.storage_context)
    
    def _init_models(self):
        """初始化模型配置"""
        self.embed_model = OllamaEmbedding(
            base_url="http://localhost:11434",
            model_name=config.get("embedding_model"),
        )

        self.llm = Ollama(
            base_url="http://localhost:11434",
            model=config.get("llm_model"),
            temperature=config.get("temperature", 0.7),
            request_timeout=300,
            gpu_layers=32  # 启用GPU加速
        )

        # 分块解析器配置
        self.node_parser = SentenceWindowNodeParser(
            window_size=512,
            window_metadata_key="context_window",
            original_text_metadata_key="original_content",
        )
        
        # 全局设置
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = config.get("chunk_size", 512)
        Settings.chunk_overlap = 128

    def _index_exists(self) -> bool:
        """验证索引文件完整性"""
        required_files = [
            "docstore.json",    # LlamaIndex的文档存储
            "faiss_index.bin",  # Faiss二进制索引文件
            "version.info"      # 版本信息文件
        ]
        
        # 检查所有必需文件是否存在
        return all(
            (self.index_dir / filename).exists() 
            for filename in required_files
        )

    def _load_existing_index(self) -> None:
        """加载已有索引的完整流程"""
        try:
            # 1. 加载版本信息
            version_file = self.index_dir / "version.info"
            with open(version_file, "r") as f:
                version_info = dict(line.strip().split(":") for line in f)
            
            # 2. 校验维度一致性
            current_embed_dim = len(self.embed_model.get_text_embedding("test"))
            if current_embed_dim != int(version_info["embed_dim"]):
                raise ValueError(
                    f"维度不匹配: 当前模型维度 {current_embed_dim} "
                    f"vs 索引维度 {version_info['embed_dim']}"
                )

            # 3. 加载Faiss索引
            faiss_index = faiss.read_index(
                str(self.index_dir / "faiss_index.bin"),
                faiss.IO_FLAG_MMAP  # 内存映射模式，适合大文件
            )
            
            # 4. 重建存储上下文
            self.vector_store = FaissVectorStore(faiss_index=faiss_index)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=self.index_dir
            )
        except Exception as e:
            print(f"❌ 索引加载失败: {str(e)}")
            print("⚠️ 正在尝试重建索引...")
            self._build_index(force_rebuild=True)

    def _init_faiss_store(self):
        """初始化Faiss存储"""
        # 定义向量维度（需与嵌入模型输出维度一致）
        test_embedding = self.embed_model.get_text_embedding("test")
        self.embed_dim = len(test_embedding)
        print("embed_model 向量维度")
        print(self.embed_dim)
        
        # 创建Faiss索引（HNSW算法）
        faiss_index = faiss.IndexHNSWFlat(self.embed_dim, 32)
        faiss_index.hnsw.efConstruction = 200
        faiss_index.hnsw.efSearch = 128
        
        # 创建向量存储
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # 更新存储上下文
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

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

    def _build_index(self, force_rebuild: bool = False) -> None:
        self._init_faiss_store()

        """构建/更新索引"""
        if not force_rebuild and (self.index_dir / "docstore.json").exists():
            print("🔄 检测到已有索引，使用缓存")
            self._load_existing_index()
            return

        print("⏳ 开始构建索引...")
        documents = self._load_documents()
        if not documents:
            raise ValueError("未找到可索引的数据文件")
        
        # 创建索引
        VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        # 持久化索引
        self._persist_faiss_index()
    
    def _persist_faiss_index(self):
        """持久化Faiss索引"""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            if not (self.index_dir / "docstore.json").exists():
                # 保存LlamaIndex元数据
                self.storage_context.persist(persist_dir=self.index_dir)
            
            # 保存Faiss索引
            faiss.write_index(
                self.vector_store.client,
                str(self.index_dir / "faiss_index.bin")
            )
            
            # 记录版本信息
            with open(self.index_dir / "version.info", "w") as f:
                f.write(f"faiss_version:{faiss.__version__}\n")
                f.write(f"embed_dim:{self.embed_dim}\n")
                
            print(f"💾 索引持久化完成（维度：{self.embed_dim}）")
            
        except Exception as e:
            print(f"❌ 持久化失败: {str(e)}")
            raise

    def get_query_engine(self, similarity_top_k: int = 5):
        """获取查询引擎"""
        if not self.index:
            raise RuntimeError("索引未正确初始化")
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="tree_summarize",
            verbose=True
        )

    @lru_cache(maxsize=1000)  # 缓存1000条查询
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


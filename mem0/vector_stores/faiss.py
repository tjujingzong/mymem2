import logging
import os
import pickle
import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

import warnings

try:
    # Suppress SWIG deprecation warnings from FAISS
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")
    
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)

    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss python package. "
        "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
        "or `pip install faiss-cpu` (depending on Python version)."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class FAISS(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        path: Optional[str] = None,
        distance_strategy: str = "euclidean",
        normalize_L2: bool = False,
        embedding_model_dims: int = 1536,
    ):
        """
        Initialize the FAISS vector store.

        Args:
            collection_name (str): Name of the collection.
            path (str, optional): Path for local FAISS database. Defaults to None.
            distance_strategy (str, optional): Distance strategy to use. Options: 'euclidean', 'inner_product', 'cosine'.
                Defaults to "euclidean".
            normalize_L2 (bool, optional): Whether to normalize L2 vectors. Only applicable for euclidean distance.
                Defaults to False.
        """
        self.collection_name = collection_name
        self.path = path or f"/tmp/faiss/{collection_name}"
        self.distance_strategy = distance_strategy
        self.normalize_L2 = normalize_L2
        self.embedding_model_dims = embedding_model_dims

        # 是否启用"按 user_id 分索引"模式（默认关闭，保持向后兼容）
        self.per_user_index = (
            str(os.getenv("MEM0_FAISS_PER_USER_INDEX", "0")).lower() in ("1", "true", "yes")
        )

        # Initialize storage structures
        self.index = None
        self.docstore = {}
        self.index_to_id = {}
        # Thread lock to protect dictionary access during save operations
        # 使用 RLock 以允许在持锁状态下安全调用 _save（避免嵌套加锁死锁）
        self._save_lock = threading.RLock()

        # 分索引模式：每个 user_id 一个独立的索引和 docstore
        if self.per_user_index:
            self.user_indices = {}  # {user_id: faiss.Index}
            self.user_docstores = {}  # {user_id: {vector_id: payload}}
            self.user_index_to_ids = {}  # {user_id: {index_id: vector_id}}
            logger.info("FAISS per-user index mode enabled")

        # Create directory if it doesn't exist
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            if self.per_user_index:
                # 分索引模式：加载所有 user 索引
                self._load_per_user_indices()
            else:
                # 原有模式：加载全局索引
                index_path = f"{self.path}/{collection_name}.faiss"
                docstore_path = f"{self.path}/{collection_name}.pkl"
                if os.path.exists(index_path) and os.path.exists(docstore_path):
                    self._load(index_path, docstore_path)
                    # If loading failed, delete corrupted files and create a new collection
                    if self.index is None:
                        logger.warning("Failed to load existing index, deleting corrupted files and creating new collection")
                        try:
                            if os.path.exists(index_path):
                                os.remove(index_path)
                            if os.path.exists(docstore_path):
                                os.remove(docstore_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete corrupted files: {e}")
                        self.create_col(collection_name)
                else:
                    self.create_col(collection_name)

    def _load(self, index_path: str, docstore_path: str):
        """
        Load FAISS index and docstore from disk.

        Args:
            index_path (str): Path to FAISS index file.
            docstore_path (str): Path to docstore pickle file.
        """
        try:
            self.index = faiss.read_index(index_path)
            with open(docstore_path, "rb") as f:
                self.docstore, self.index_to_id = pickle.load(f)
            logger.info(f"Loaded FAISS index from {index_path} with {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            # 确保在加载失败时，将index设置为None，以便后续创建新集合
            self.index = None
            self.docstore = {}
            self.index_to_id = {}

    def _load_per_user_indices(self):
        """加载所有 user 级别的索引（分索引模式）"""
        if not self.path:
            return

        try:
            user_indices_path = f"{self.path}/{self.collection_name}_user_indices.pkl"
            if os.path.exists(user_indices_path):
                with open(user_indices_path, "rb") as f:
                    user_data = pickle.load(f)
                    self.user_docstores = user_data.get("docstores", {})
                    self.user_index_to_ids = user_data.get("index_to_ids", {})
                
                # 加载每个 user 的 FAISS 索引文件
                self.user_indices = {}
                for user_id in self.user_docstores.keys():
                    user_index_path = f"{self.path}/{self.collection_name}_user_{user_id}.faiss"
                    if os.path.exists(user_index_path):
                        try:
                            self.user_indices[user_id] = faiss.read_index(user_index_path)
                        except Exception as e:
                            logger.warning(f"Failed to load index for user_id={user_id}: {e}")
                            # 如果索引文件损坏，清空对应的 docstore
                            self.user_docstores[user_id] = {}
                            self.user_index_to_ids[user_id] = {}
                
                logger.info(f"Loaded {len(self.user_indices)} user indices from {user_indices_path}")
            else:
                logger.info("No existing per-user indices found, will create new ones")
        except Exception as e:
            logger.warning(f"Failed to load per-user indices: {e}")
            self.user_indices = {}
            self.user_docstores = {}
            self.user_index_to_ids = {}

    def _save(self):
        """Save FAISS index and docstore to disk."""
        if not self.path:
            return

        # Use lock to prevent concurrent modifications during save
        with self._save_lock:
            try:
                os.makedirs(self.path, exist_ok=True)
                
                if self.per_user_index:
                    # 分索引模式：保存所有 user 索引
                    user_indices_path = f"{self.path}/{self.collection_name}_user_indices.pkl"
                    # FAISS 索引不能直接 pickle，需要单独保存每个索引文件
                    # 这里只保存 docstores 和 index_to_ids 的映射
                    user_data = {
                        "indices": {},  # FAISS 索引需要单独保存
                        "docstores": deepcopy(self.user_docstores),
                        "index_to_ids": deepcopy(self.user_index_to_ids),
                    }
                    with open(user_indices_path, "wb") as f:
                        pickle.dump(user_data, f)
                    
                    # 保存每个 user 的 FAISS 索引文件
                    for user_id, index in self.user_indices.items():
                        if index and index.ntotal > 0:
                            user_index_path = f"{self.path}/{self.collection_name}_user_{user_id}.faiss"
                            faiss.write_index(index, user_index_path)
                    
                    logger.debug(f"Saved {len(self.user_indices)} user indices")
                else:
                    # 原有模式：保存全局索引
                    if not self.index:
                        return
                    index_path = f"{self.path}/{self.collection_name}.faiss"
                    docstore_path = f"{self.path}/{self.collection_name}.pkl"

                    faiss.write_index(self.index, index_path)
                    # Create deep copies of dictionaries to avoid "dictionary changed size during iteration" error
                    # This prevents issues when dictionaries are modified by other threads during serialization
                    docstore_copy = deepcopy(self.docstore)
                    index_to_id_copy = deepcopy(self.index_to_id)
                    with open(docstore_path, "wb") as f:
                        pickle.dump((docstore_copy, index_to_id_copy), f)
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

    def _parse_output(self, scores, ids, limit=None, user_id=None) -> List[OutputData]:
        """
        Parse the output data.

        Args:
            scores: Similarity scores from FAISS.
            ids: Indices from FAISS.
            limit: Maximum number of results to return.
            user_id: Optional user_id for per-user index mode.

        Returns:
            List[OutputData]: Parsed output data.
        """
        if limit is None:
            limit = len(ids)

        results = []
        # Use lock to protect dictionary access during iteration
        with self._save_lock:
            # 分索引模式：使用 user 级别的 docstore 和 index_to_id
            if self.per_user_index and user_id:
                docstore = self.user_docstores.get(user_id, {})
                index_to_id = self.user_index_to_ids.get(user_id, {})
            else:
                docstore = self.docstore
                index_to_id = self.index_to_id

            for i in range(min(len(ids), limit)):
                if ids[i] == -1:  # FAISS returns -1 for empty results
                    continue

                index_id = int(ids[i])
                vector_id = index_to_id.get(index_id)
                if vector_id is None:
                    continue

                payload = docstore.get(vector_id)
                if payload is None:
                    continue

                payload_copy = payload.copy()

                score = float(scores[i])
                entry = OutputData(
                    id=vector_id,
                    score=score,
                    payload=payload_copy,
                )
                results.append(entry)

        return results

    def create_col(self, name: str, distance: str = None):
        """
        Create a new collection.

        Args:
            name (str): Name of the collection.
            distance (str, optional): Distance metric to use. Overrides the distance_strategy
                passed during initialization. Defaults to None.

        Returns:
            self: The FAISS instance.
        """
        distance_strategy = distance or self.distance_strategy

        # Create index based on distance strategy
        if distance_strategy.lower() == "inner_product" or distance_strategy.lower() == "cosine":
            index_class = faiss.IndexFlatIP
        else:
            index_class = faiss.IndexFlatL2

        if self.per_user_index:
            # 分索引模式：不创建全局索引，按需为每个 user 创建
            self.collection_name = name
            logger.info("Per-user index mode: will create indices on-demand for each user")
        else:
            # 原有模式：创建全局索引
            self.index = index_class(self.embedding_model_dims)
            self.collection_name = name
            self._save()

        return self

    def _get_or_create_user_index(self, user_id: str):
        """获取或创建指定 user_id 的索引（分索引模式）"""
        if user_id not in self.user_indices:
            # 创建新的 user 索引
            if self.distance_strategy.lower() == "inner_product" or self.distance_strategy.lower() == "cosine":
                self.user_indices[user_id] = faiss.IndexFlatIP(self.embedding_model_dims)
            else:
                self.user_indices[user_id] = faiss.IndexFlatL2(self.embedding_model_dims)
            
            # 初始化对应的 docstore 和 index_to_id
            if user_id not in self.user_docstores:
                self.user_docstores[user_id] = {}
            if user_id not in self.user_index_to_ids:
                self.user_index_to_ids[user_id] = {}
            
            logger.debug(f"Created new index for user_id={user_id}")
        
        return self.user_indices[user_id]

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Insert vectors into a collection.

        Args:
            vectors (List[list]): List of vectors to insert.
            payloads (Optional[List[Dict]], optional): List of payloads corresponding to vectors. Defaults to None.
            ids (Optional[List[str]], optional): List of IDs corresponding to vectors. Defaults to None.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]

        if len(vectors) != len(ids) or len(vectors) != len(payloads):
            raise ValueError("Vectors, payloads, and IDs must have the same length")

        vectors_np = np.array(vectors, dtype=np.float32)

        if self.normalize_L2 and self.distance_strategy.lower() == "euclidean":
            faiss.normalize_L2(vectors_np)

        if self.per_user_index:
            # 分索引模式：按 user_id 分组插入
            user_groups = {}
            for i, payload in enumerate(payloads):
                user_id = payload.get("user_id")
                if not user_id:
                    logger.warning(f"Vector {ids[i]} has no user_id, skipping")
                    continue
                if user_id not in user_groups:
                    user_groups[user_id] = {"vectors": [], "ids": [], "payloads": []}
                user_groups[user_id]["vectors"].append(vectors_np[i])
                user_groups[user_id]["ids"].append(ids[i])
                user_groups[user_id]["payloads"].append(payload)

            # 为每个 user 插入到对应的索引
            with self._save_lock:
                for user_id, group in user_groups.items():
                    user_index = self._get_or_create_user_index(user_id)
                    user_vectors = np.array(group["vectors"], dtype=np.float32)
                    starting_idx = user_index.ntotal
                    user_index.add(user_vectors)

                    for j, (vector_id, payload) in enumerate(zip(group["ids"], group["payloads"])):
                        self.user_docstores[user_id][vector_id] = payload.copy()
                        self.user_index_to_ids[user_id][starting_idx + j] = vector_id

            self._save()
            logger.info(f"Inserted {len(vectors)} vectors into per-user indices ({len(user_groups)} users)")
        else:
            # 原有模式：插入到全局索引
            if self.index is None:
                raise ValueError("Collection not initialized. Call create_col first.")

            starting_idx = self.index.ntotal
            self.index.add(vectors_np)

            # Use lock to protect dictionary modifications
            with self._save_lock:
                for i, (vector_id, payload) in enumerate(zip(ids, payloads)):
                    self.docstore[vector_id] = payload.copy()
                    self.index_to_id[starting_idx + i] = vector_id

            self._save()
            logger.info(f"Inserted {len(vectors)} vectors into collection {self.collection_name}")

    def search(
        self, query: str, vectors: List[list], limit: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """
        Search for similar vectors.

        Args:
            query (str): Query (not used, kept for API compatibility).
            vectors (List[list]): List of vectors to search.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (Optional[Dict], optional): Filters to apply to the search. Defaults to None.

        Returns:
            List[OutputData]: Search results.
        """
        query_vectors = np.array(vectors, dtype=np.float32)

        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)

        if self.normalize_L2 and self.distance_strategy.lower() == "euclidean":
            faiss.normalize_L2(query_vectors)

        if self.per_user_index and filters and "user_id" in filters:
            # 分索引模式：只在指定 user_id 的索引上搜索
            user_id = filters["user_id"]
            user_index = self.user_indices.get(user_id)
            
            if user_index is None or user_index.ntotal == 0:
                # 该 user 还没有索引或索引为空
                logger.debug(f"No index found for user_id={user_id}, returning empty results")
                return []
            
            # 直接在该 user 的索引上搜索，不需要 fetch_k 放大，因为已经是该 user 的全部向量
            scores, indices = user_index.search(query_vectors, limit)
            results = self._parse_output(scores[0], indices[0], limit, user_id=user_id)
            
            # 仍然应用其他 filters（如果有）
            if filters:
                filtered_results = []
                for result in results:
                    if self._apply_filters(result.payload, filters):
                        filtered_results.append(result)
                        if len(filtered_results) >= limit:
                            break
                results = filtered_results[:limit]
            
            return results
        else:
            # 原有模式：在全局索引上搜索
            if self.index is None:
                raise ValueError("Collection not initialized. Call create_col first.")

            fetch_k = limit * 2 if filters else limit
            scores, indices = self.index.search(query_vectors, fetch_k)

            results = self._parse_output(scores[0], indices[0], limit)

            if filters:
                filtered_results = []
                for result in results:
                    if self._apply_filters(result.payload, filters):
                        filtered_results.append(result)
                        if len(filtered_results) >= limit:
                            break
                results = filtered_results[:limit]

            return results

    def _apply_filters(self, payload: Dict, filters: Dict) -> bool:
        """
        Apply filters to a payload.

        Args:
            payload (Dict): Payload to filter.
            filters (Dict): Filters to apply.

        Returns:
            bool: True if payload passes filters, False otherwise.
        """
        if not filters or not payload:
            return True

        for key, value in filters.items():
            if key not in payload:
                return False

            if isinstance(value, list):
                if payload[key] not in value:
                    return False
            elif payload[key] != value:
                return False

        return True

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        if self.per_user_index:
            # 分索引模式：需要找到该 vector_id 属于哪个 user
            with self._save_lock:
                for user_id, docstore in self.user_docstores.items():
                    if vector_id in docstore:
                        index_to_id = self.user_index_to_ids.get(user_id, {})
                        index_to_delete = None
                        for idx, vid in index_to_id.items():
                            if vid == vector_id:
                                index_to_delete = idx
                                break
                        
                        if index_to_delete is not None:
                            docstore.pop(vector_id, None)
                            index_to_id.pop(index_to_delete, None)
                            self._save()
                            logger.info(f"Deleted vector {vector_id} from user_id={user_id}")
                        return
                logger.warning(f"Vector {vector_id} not found in any user index")
        else:
            # 原有模式
            if self.index is None:
                raise ValueError("Collection not initialized. Call create_col first.")

            index_to_delete = None
            # Use lock to protect dictionary access during iteration
            with self._save_lock:
                for idx, vid in self.index_to_id.items():
                    if vid == vector_id:
                        index_to_delete = idx
                        break

            if index_to_delete is not None:
                # Use lock to protect dictionary modifications
                with self._save_lock:
                    self.docstore.pop(vector_id, None)
                    self.index_to_id.pop(index_to_delete, None)

                self._save()

                logger.info(f"Deleted vector {vector_id} from collection {self.collection_name}")
            else:
                logger.warning(f"Vector {vector_id} not found in collection {self.collection_name}")

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ):
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update.
            vector (Optional[List[float]], optional): Updated vector. Defaults to None.
            payload (Optional[Dict], optional): Updated payload. Defaults to None.
        """
        if self.index is None:
            raise ValueError("Collection not initialized. Call create_col first.")

        # Use lock to protect dictionary access
        with self._save_lock:
            if vector_id not in self.docstore:
                raise ValueError(f"Vector {vector_id} not found")

            current_payload = self.docstore[vector_id].copy()

            if payload is not None:
                self.docstore[vector_id] = payload.copy()
                current_payload = self.docstore[vector_id].copy()

        if vector is not None:
            self.delete(vector_id)
            self.insert([vector], [current_payload], [vector_id])
        else:
            self._save()

        logger.info(f"Updated vector {vector_id} in collection {self.collection_name}")

    def get(self, vector_id: str) -> OutputData:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            OutputData: Retrieved vector.
        """
        if self.per_user_index:
            # 分索引模式：在所有 user 的 docstore 中查找
            with self._save_lock:
                for user_id, docstore in self.user_docstores.items():
                    if vector_id in docstore:
                        payload = docstore[vector_id].copy()
                        return OutputData(
                            id=vector_id,
                            score=None,
                            payload=payload,
                        )
                return None
        else:
            # 原有模式
            if self.index is None:
                raise ValueError("Collection not initialized. Call create_col first.")

            # Use lock to protect dictionary access
            with self._save_lock:
                if vector_id not in self.docstore:
                    return None

                payload = self.docstore[vector_id].copy()

            return OutputData(
                id=vector_id,
                score=None,
                payload=payload,
            )

    def list_cols(self) -> List[str]:
        """
        List all collections.

        Returns:
            List[str]: List of collection names.
        """
        if not self.path:
            return [self.collection_name] if self.index else []

        try:
            collections = []
            path = Path(self.path).parent
            for file in path.glob("*.faiss"):
                collections.append(file.stem)
            return collections
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            return [self.collection_name] if self.index else []

    def delete_col(self):
        """
        Delete a collection.
        """
        if self.path:
            try:
                index_path = f"{self.path}/{self.collection_name}.faiss"
                docstore_path = f"{self.path}/{self.collection_name}.pkl"

                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(docstore_path):
                    os.remove(docstore_path)

                logger.info(f"Deleted collection {self.collection_name}")
            except Exception as e:
                logger.warning(f"Failed to delete collection: {e}")

        self.index = None
        self.docstore = {}
        self.index_to_id = {}

    def col_info(self) -> Dict:
        """
        Get information about a collection.

        Returns:
            Dict: Collection information.
        """
        if self.index is None:
            return {"name": self.collection_name, "count": 0}

        return {
            "name": self.collection_name,
            "count": self.index.ntotal,
            "dimension": self.index.d,
            "distance": self.distance_strategy,
        }

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[OutputData]:
        """
        List all vectors in a collection.

        Args:
            filters (Optional[Dict], optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            List[OutputData]: List of vectors.
        """
        results = []
        count = 0

        # Use lock to protect dictionary iteration
        with self._save_lock:
            if self.per_user_index:
                # 分索引模式：遍历所有 user 的 docstore
                target_user_id = filters.get("user_id") if filters else None
                docstores_to_search = (
                    {target_user_id: self.user_docstores[target_user_id]} 
                    if target_user_id and target_user_id in self.user_docstores
                    else self.user_docstores
                )
                
                for user_id, docstore in docstores_to_search.items():
                    for vector_id, payload in docstore.items():
                        if filters and not self._apply_filters(payload, filters):
                            continue

                        payload_copy = payload.copy()

                        results.append(
                            OutputData(
                                id=vector_id,
                                score=None,
                                payload=payload_copy,
                            )
                        )

                        count += 1
                        if count >= limit:
                            break
                    
                    if count >= limit:
                        break
            else:
                # 原有模式
                if self.index is None:
                    return []

                for vector_id, payload in self.docstore.items():
                    if filters and not self._apply_filters(payload, filters):
                        continue

                    payload_copy = payload.copy()

                    results.append(
                        OutputData(
                            id=vector_id,
                            score=None,
                            payload=payload_copy,
                        )
                    )

                    count += 1
                    if count >= limit:
                        break

        return [results]

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.collection_name)

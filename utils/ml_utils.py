"""
Utilitários para gestão de modelos ML, cache e otimização de recursos.

Este módulo gerencia o carregamento, cache e otimização de modelos de machine learning
para análise semântica, garantindo uso eficiente de recursos computacionais.
"""

import os
import gc
import time
import psutil
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache, wraps
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import warnings
import os

# Suprime warnings desnecessários dos modelos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Fix para problema de torch com _unsafe_update_src
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.language_models import LanguageModels, ModelConfig, MODEL_CACHE


class ModelManager:
    """Gerenciador de modelos ML com cache inteligente."""
    
    def __init__(self, max_memory_usage: float = 0.8):
        """
        Inicializa o gerenciador de modelos.
        
        Args:
            max_memory_usage: Percentual máximo de memória a ser utilizado (0.0-1.0)
        """
        self.max_memory_usage = max_memory_usage
        self.loaded_models = {}
        self.model_usage_stats = {}
        self.device = self._get_optimal_device()
    
    def _get_optimal_device(self) -> str:
        """Determina o dispositivo ótimo (CPU/GPU)."""
        # Força CPU se houver problemas com CUDA
        if os.getenv("FORCE_CPU", "false").lower() == "true":
            return "cpu"
            
        if torch.cuda.is_available():
            try:
                # Testa se CUDA está funcionando corretamente
                torch.cuda.empty_cache()
                return "cuda"
            except Exception:
                print("⚠️ CUDA disponível mas com problemas, usando CPU")
                return "cpu"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _check_memory_usage(self) -> float:
        """Verifica o uso atual de memória."""
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    
    def _should_load_model(self, model_name: str) -> bool:
        """Verifica se é seguro carregar um novo modelo."""
        current_memory = self._check_memory_usage()
        if current_memory > self.max_memory_usage:
            self._cleanup_unused_models()
            current_memory = self._check_memory_usage()
        
        return current_memory < self.max_memory_usage
    
    def _cleanup_unused_models(self):
        """Remove modelos menos utilizados da memória."""
        if not self.model_usage_stats:
            return
        
        # Ordena por uso (menos usado primeiro)
        sorted_models = sorted(
            self.model_usage_stats.items(), 
            key=lambda x: (x[1]['last_used'], x[1]['usage_count'])
        )
        
        # Remove até 50% dos modelos menos utilizados
        models_to_remove = len(sorted_models) // 2
        
        for model_name, _ in sorted_models[:models_to_remove]:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                print(f"🧹 Modelo removido da memória: {model_name}")
        
        # Força coleta de lixo
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def load_model(self, config: ModelConfig) -> Any:
        """
        Carrega um modelo com cache inteligente.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Modelo carregado e pronto para uso
        """
        model_key = f"{config.name}_{config.model_id}"
        
        # Verifica se já está carregado
        if model_key in self.loaded_models:
            self._update_usage_stats(model_key)
            return self.loaded_models[model_key]
        
        # Verifica memória antes de carregar
        if not self._should_load_model(model_key):
            raise RuntimeError(f"Memória insuficiente para carregar modelo: {config.name}")
        
        print(f"🔄 Carregando modelo: {config.name}")
        start_time = time.time()
        
        try:
            # Carrega modelo baseado na tarefa
            if config.task == "feature-extraction" and "sentence-transformers" in config.model_id:
                model = SentenceTransformer(config.model_id, device=self.device)
            else:
                model = pipeline(
                    config.task,
                    model=config.model_id,
                    device=0 if self.device == "cuda" else -1,
                    max_length=config.max_length,
                    truncation=True
                )
            
            load_time = time.time() - start_time
            
            # Armazena no cache
            self.loaded_models[model_key] = model
            self._init_usage_stats(model_key, load_time)
            
            print(f"✅ Modelo carregado: {config.name} ({load_time:.2f}s)")
            return model
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {config.name}: {e}")
            raise
    
    def _update_usage_stats(self, model_key: str):
        """Atualiza estatísticas de uso do modelo."""
        if model_key in self.model_usage_stats:
            self.model_usage_stats[model_key]['usage_count'] += 1
            self.model_usage_stats[model_key]['last_used'] = time.time()
    
    def _init_usage_stats(self, model_key: str, load_time: float):
        """Inicializa estatísticas de uso para um novo modelo."""
        self.model_usage_stats[model_key] = {
            'usage_count': 1,
            'last_used': time.time(),
            'load_time': load_time,
            'created_at': time.time()
        }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos modelos carregados."""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'memory_usage': self._check_memory_usage(),
            'device': self.device,
            'usage_stats': self.model_usage_stats
        }
    
    def unload_all_models(self):
        """Remove todos os modelos da memória."""
        self.loaded_models.clear()
        self.model_usage_stats.clear()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("🧹 Todos os modelos removidos da memória")


# Instância global do gerenciador
model_manager = ModelManager()


def get_model_for_language_and_task(language: str, task: str) -> Any:
    """
    Obtém modelo apropriado para idioma e tarefa específicos.
    
    Args:
        language: Código do idioma (pt, en, etc.)
        task: Tipo de tarefa (sentiment, embeddings, emotion)
        
    Returns:
        Modelo carregado
    """
    models = LanguageModels.get_models_for_language(language)
    
    if task not in models:
        raise ValueError(f"Tarefa '{task}' não suportada para idioma '{language}'")
    
    config = models[task]
    return model_manager.load_model(config)


@lru_cache(maxsize=128)
def detect_language(text: str) -> str:
    """
    Detecta idioma do texto com cache.
    
    Args:
        text: Texto para análise
        
    Returns:
        Código do idioma detectado
    """
    try:
        # Carrega detector de idioma
        detector = model_manager.load_model(LanguageModels.LANGUAGE_DETECTION)
        
        # Detecta idioma
        result = detector(text[:512])  # Limita texto para performance
        
        if isinstance(result, list) and len(result) > 0:
            detected = result[0]['label'].lower()
            # Mapeia códigos específicos
            language_map = {
                'portuguese': 'pt',
                'english': 'en',
                'spanish': 'es'
            }
            return language_map.get(detected, detected[:2])
        
        return 'en'  # Fallback
        
    except Exception as e:
        print(f"⚠️ Erro na detecção de idioma: {e}")
        return 'en'  # Fallback seguro


def batch_process(func):
    """
    Decorator para processamento em lotes otimizado.
    
    Args:
        func: Função a ser decorada
        
    Returns:
        Função decorada com suporte a batches
    """
    @wraps(func)
    def wrapper(texts: Union[str, List[str]], batch_size: int = 8, **kwargs):
        # Converte string única em lista
        if isinstance(texts, str):
            return func([texts], **kwargs)[0]
        
        # Processa em lotes
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = func(batch, **kwargs)
            results.extend(batch_results)
        
        return results
    
    return wrapper


def get_text_embeddings(texts: List[str], language: str = "pt") -> List[List[float]]:
    """
    Gera embeddings semânticos para lista de textos.
    
    Args:
        texts: Lista de textos
        language: Código do idioma
        
    Returns:
        Lista de embeddings
    """
    model = get_model_for_language_and_task(language, "embeddings")
    
    try:
        if isinstance(model, SentenceTransformer):
            embeddings = model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        else:
            # Fallback: processa um por vez
            results = []
            for text in texts:
                try:
                    result = model(text)
                    if isinstance(result, dict) and 'embeddings' in result:
                        results.append(result['embeddings'])
                    else:
                        # Gera embedding dummy se falhar
                        results.append([0.0] * 384)  # Tamanho padrão
                except:
                    results.append([0.0] * 384)
            return results
    except Exception as e:
        # Fallback completo: embeddings dummy
        print(f"⚠️ Erro nos embeddings, usando fallback: {e}")
        return [[0.0] * 384 for _ in texts]


def analyze_sentiment(texts: List[str], language: str = "pt") -> List[Dict[str, Any]]:
    """
    Analisa sentimentos de lista de textos.
    
    Args:
        texts: Lista de textos
        language: Código do idioma
        
    Returns:
        Lista de análises de sentimento
    """
    model = get_model_for_language_and_task(language, "sentiment")
    
    # Processa cada texto individualmente para evitar erros de batch
    normalized_results = []
    for text in texts:
        try:
            result = model(text)
            if isinstance(result, list):
                result = result[0]  # Pega primeiro resultado se for lista
            
            normalized_results.append({
                'label': result['label'],
                'score': result['score'],
                'confidence': result['score']
            })
        except Exception as e:
            # Fallback em caso de erro
            normalized_results.append({
                'label': 'neutral',
                'score': 0.5,
                'confidence': 0.5
            })
    
    return normalized_results


def analyze_emotions(texts: List[str], language: str = "pt") -> List[Dict[str, Any]]:
    """
    Analisa emoções de lista de textos.
    
    Args:
        texts: Lista de textos
        language: Código do idioma
        
    Returns:
        Lista de análises de emoção
    """
    model = get_model_for_language_and_task(language, "emotion")
    
    # Processa cada texto individualmente para evitar erros de batch
    normalized_results = []
    for text in texts:
        try:
            result = model(text)
            if isinstance(result, list):
                result = result[0]  # Pega primeiro resultado se for lista
            
            normalized_results.append({
                'emotion': result['label'],
                'confidence': result['score'],
                'all_emotions': result if 'score' in result else {}
            })
        except Exception as e:
            # Fallback em caso de erro
            normalized_results.append({
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {}
            })
    
    return normalized_results


def optimize_memory_usage():
    """Otimiza uso de memória removendo modelos desnecessários."""
    model_manager._cleanup_unused_models()


def get_system_info() -> Dict[str, Any]:
    """
    Retorna informações do sistema para debugging.
    
    Returns:
        Dicionário com informações do sistema
    """
    memory = psutil.virtual_memory()
    
    info = {
        'memory_total_gb': memory.total / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
        'cpu_count': psutil.cpu_count(),
        'device': model_manager.device,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
    
    return info


def preload_models_for_language(language: str):
    """
    Pré-carrega todos os modelos para um idioma específico.
    
    Args:
        language: Código do idioma
    """
    print(f"🔄 Pré-carregando modelos para idioma: {language}")
    
    tasks = ["sentiment", "embeddings", "emotion"]
    for task in tasks:
        try:
            get_model_for_language_and_task(language, task)
        except Exception as e:
            print(f"⚠️ Erro ao pré-carregar {task} para {language}: {e}")
    
    print(f"✅ Modelos pré-carregados para {language}")
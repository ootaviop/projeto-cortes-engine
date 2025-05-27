"""
Configuração de modelos de ML por idioma para análise semântica multilíngue.

Este módulo centraliza a configuração de todos os modelos de machine learning
utilizados para análise semântica, organizados por idioma e tarefa específica.
Permite fácil expansão para novos idiomas e culturas.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuração de um modelo específico."""
    name: str
    model_id: str
    task: str
    confidence_threshold: float = 0.7
    max_length: int = 512
    batch_size: int = 8


class LanguageModels:
    """Configuração centralizada de modelos por idioma."""
    
    # Detecção de idioma
    LANGUAGE_DETECTION = ModelConfig(
        name="language_detector",
        model_id="papluca/xlm-roberta-base-language-detection",
        task="text-classification",
        confidence_threshold=0.8
    )
    
    # Modelos para Português Brasileiro
    PORTUGUESE_MODELS = {
        "sentiment": ModelConfig(
            name="pt_sentiment",
            model_id="neuralmind/bert-base-portuguese-cased",
            task="sentiment-analysis",
            confidence_threshold=0.75
        ),
        "embeddings": ModelConfig(
            name="pt_embeddings", 
            model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            task="feature-extraction",
            max_length=384
        ),
        "emotion": ModelConfig(
            name="pt_emotion",
            model_id="neuralmind/bert-base-portuguese-cased",
            task="text-classification",
            confidence_threshold=0.6
        )
    }
    
    # Modelos para Inglês
    ENGLISH_MODELS = {
        "sentiment": ModelConfig(
            name="en_sentiment",
            model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
            task="sentiment-analysis",
            confidence_threshold=0.8
        ),
        "embeddings": ModelConfig(
            name="en_embeddings",
            model_id="sentence-transformers/all-MiniLM-L6-v2", 
            task="feature-extraction",
            max_length=256
        ),
        "emotion": ModelConfig(
            name="en_emotion",
            model_id="j-hartmann/emotion-english-distilroberta-base",
            task="text-classification",
            confidence_threshold=0.7
        )
    }
    
    # Modelos Multilíngues (fallback)
    MULTILINGUAL_MODELS = {
        "sentiment": ModelConfig(
            name="multi_sentiment",
            model_id="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            task="sentiment-analysis",
            confidence_threshold=0.7
        ),
        "embeddings": ModelConfig(
            name="multi_embeddings",
            model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
            task="feature-extraction",
            max_length=512
        ),
        "emotion": ModelConfig(
            name="multi_emotion", 
            model_id="cardiffnlp/twitter-xlm-roberta-base-emotion",
            task="text-classification",
            confidence_threshold=0.65
        )
    }
    
    @classmethod
    def get_models_for_language(cls, language_code: str) -> Dict[str, ModelConfig]:
        """
        Retorna modelos apropriados para um idioma específico.
        
        Args:
            language_code: Código do idioma (pt, en, es, etc.)
            
        Returns:
            Dicionário com modelos configurados para o idioma
        """
        language_map = {
            "pt": cls.PORTUGUESE_MODELS,
            "en": cls.ENGLISH_MODELS,
        }
        
        # Retorna modelos específicos ou multilíngues como fallback
        return language_map.get(language_code, cls.MULTILINGUAL_MODELS)
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Retorna lista de idiomas com suporte específico."""
        return ["pt", "en"]
    
    @classmethod
    def is_language_supported(cls, language_code: str) -> bool:
        """Verifica se um idioma tem suporte específico."""
        return language_code in cls.get_supported_languages()


# Configurações específicas por contexto cultural
CULTURAL_CONFIGS = {
    "pt": {
        "intensifiers": ["muito", "demais", "pra caramba", "absurdo", "inacreditável"],
        "sarcasm_indicators": ["né", "claro", "obviamente", "com certeza"],
        "excitement_markers": ["cara", "mano", "nossa", "caramba", "eita"],
        "emotion_multiplier": 1.2,  # Brasileiros são mais expressivos
        "confidence_adjustment": 0.1  # Ajuste para expressões culturais
    },
    "en": {
        "intensifiers": ["very", "extremely", "absolutely", "incredibly", "amazing"],
        "sarcasm_indicators": ["obviously", "sure", "right", "yeah right"],
        "excitement_markers": ["wow", "amazing", "incredible", "awesome", "dude"],
        "emotion_multiplier": 1.0,
        "confidence_adjustment": 0.0
    }
}


def get_cultural_config(language_code: str) -> Dict[str, Any]:
    """
    Retorna configurações culturais para um idioma específico.
    
    Args:
        language_code: Código do idioma
        
    Returns:
        Dicionário com configurações culturais
    """
    return CULTURAL_CONFIGS.get(language_code, CULTURAL_CONFIGS["en"])


# Cache de modelos carregados (será usado por ml_utils.py)
MODEL_CACHE = {}
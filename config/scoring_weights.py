"""
Configuração de pesos e thresholds para scoring de momentos virais.

Este módulo define os pesos utilizados no sistema de scoring multi-dimensional
para identificar e ranquear momentos com potencial viral em diferentes tipos
de conteúdo e contextos culturais.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ScoringDimensions:
    """Dimensões do sistema de scoring."""
    emotional_intensity: float
    semantic_novelty: float
    surprise_potential: float
    information_density: float
    narrative_completeness: float
    engagement_potential: float


class ScoringWeights:
    """Configuração de pesos para diferentes contextos."""
    
    # Pesos padrão para conteúdo geral
    DEFAULT_WEIGHTS = ScoringDimensions(
        emotional_intensity=0.25,    # 25% - Picos emocionais
        semantic_novelty=0.20,       # 20% - Novidade do conteúdo
        surprise_potential=0.20,     # 20% - Potencial de surpresa
        information_density=0.15,    # 15% - Densidade informacional
        narrative_completeness=0.10, # 10% - História completa
        engagement_potential=0.10    # 10% - Padrões de engajamento
    )
    
    # Pesos para podcasts/entrevistas (foco em revelações e insights)
    PODCAST_WEIGHTS = ScoringDimensions(
        emotional_intensity=0.15,
        semantic_novelty=0.30,       # Maior peso para novidades
        surprise_potential=0.25,     # Revelações inesperadas
        information_density=0.20,    # Conteúdo denso
        narrative_completeness=0.05,
        engagement_potential=0.05
    )
    
    # Pesos para vlogs/conteúdo pessoal (foco em emoção e conexão)
    VLOG_WEIGHTS = ScoringDimensions(
        emotional_intensity=0.35,    # Maior peso emocional
        semantic_novelty=0.15,
        surprise_potential=0.20,
        information_density=0.10,
        narrative_completeness=0.15, # Histórias pessoais
        engagement_potential=0.05
    )
    
    # Pesos para conteúdo educacional (foco em clareza e completude)
    EDUCATIONAL_WEIGHTS = ScoringDimensions(
        emotional_intensity=0.10,
        semantic_novelty=0.25,
        surprise_potential=0.15,
        information_density=0.30,    # Máxima densidade informacional
        narrative_completeness=0.15,
        engagement_potential=0.05
    )
    
    # Pesos para debates/discussões (foco em conflito e argumentação)
    DEBATE_WEIGHTS = ScoringDimensions(
        emotional_intensity=0.30,    # Tensão emocional
        semantic_novelty=0.20,
        surprise_potential=0.25,     # Argumentos inesperados
        information_density=0.15,
        narrative_completeness=0.05,
        engagement_potential=0.05
    )


# Thresholds para filtros de qualidade
QUALITY_THRESHOLDS = {
    "minimum_score": 0.6,           # Score mínimo para considerar o momento
    "minimum_duration": 10.0,       # Duração mínima em segundos
    "maximum_duration": 90.0,       # Duração máxima em segundos
    "optimal_duration_min": 15.0,   # Duração ótima mínima
    "optimal_duration_max": 60.0,   # Duração ótima máxima
    "silence_threshold": 3.0,       # Máximo de silêncio permitido (segundos)
    "overlap_threshold": 0.3        # Máximo de sobreposição entre cortes (30%)
}

# Configurações por idioma/cultura
CULTURAL_SCORING_ADJUSTMENTS = {
    "pt": {
        "emotion_boost": 1.15,          # Brasileiros são mais expressivos
        "surprise_sensitivity": 1.10,   # Maior reação a surpresas
        "humor_detection_boost": 1.20,  # Cultura do humor brasileiro
        "sarcasm_penalty": 0.95,        # Sarcasmo pode ser mal interpretado
        "intensity_threshold": 0.65     # Threshold ajustado para expressividade
    },
    "en": {
        "emotion_boost": 1.0,
        "surprise_sensitivity": 1.0,
        "humor_detection_boost": 1.0,
        "sarcasm_penalty": 1.0,
        "intensity_threshold": 0.7
    }
}

# Configurações de duração por tipo de conteúdo
DURATION_CONFIGS = {
    "podcast": {
        "min_duration": 20.0,
        "max_duration": 120.0,
        "optimal_range": (30.0, 90.0)
    },
    "vlog": {
        "min_duration": 10.0,
        "max_duration": 60.0,
        "optimal_range": (15.0, 45.0)
    },
    "educational": {
        "min_duration": 15.0,
        "max_duration": 180.0,
        "optimal_range": (30.0, 120.0)
    },
    "debate": {
        "min_duration": 15.0,
        "max_duration": 90.0,
        "optimal_range": (20.0, 60.0)
    },
    "default": {
        "min_duration": 10.0,
        "max_duration": 90.0,
        "optimal_range": (15.0, 60.0)
    }
}


def get_weights_for_content_type(content_type: str) -> ScoringDimensions:
    """
    Retorna pesos apropriados para um tipo de conteúdo específico.
    
    Args:
        content_type: Tipo de conteúdo (podcast, vlog, educational, debate, default)
        
    Returns:
        ScoringDimensions com pesos configurados
    """
    weight_map = {
        "podcast": ScoringWeights.PODCAST_WEIGHTS,
        "vlog": ScoringWeights.VLOG_WEIGHTS,
        "educational": ScoringWeights.EDUCATIONAL_WEIGHTS,
        "debate": ScoringWeights.DEBATE_WEIGHTS
    }
    
    return weight_map.get(content_type, ScoringWeights.DEFAULT_WEIGHTS)


def get_cultural_adjustments(language_code: str) -> Dict[str, float]:
    """
    Retorna ajustes culturais para um idioma específico.
    
    Args:
        language_code: Código do idioma (pt, en, etc.)
        
    Returns:
        Dicionário com ajustes culturais
    """
    return CULTURAL_SCORING_ADJUSTMENTS.get(language_code, CULTURAL_SCORING_ADJUSTMENTS["en"])


def get_duration_config(content_type: str) -> Dict[str, Any]:
    """
    Retorna configuração de duração para um tipo de conteúdo.
    
    Args:
        content_type: Tipo de conteúdo
        
    Returns:
        Dicionário com configurações de duração
    """
    return DURATION_CONFIGS.get(content_type, DURATION_CONFIGS["default"])


def calculate_duration_score(duration: float, content_type: str = "default") -> float:
    """
    Calcula score baseado na duração do segmento.
    
    Args:
        duration: Duração em segundos
        content_type: Tipo de conteúdo
        
    Returns:
        Score de duração (0.0 a 1.0)
    """
    config = get_duration_config(content_type)
    optimal_min, optimal_max = config["optimal_range"]
    
    # Duração ideal recebe score máximo
    if optimal_min <= duration <= optimal_max:
        return 1.0
    
    # Penalização gradual para durações fora do ideal
    if duration < optimal_min:
        min_duration = config["min_duration"]
        if duration < min_duration:
            return 0.0
        return (duration - min_duration) / (optimal_min - min_duration)
    
    else:  # duration > optimal_max
        max_duration = config["max_duration"]
        if duration > max_duration:
            return 0.0
        return 1.0 - ((duration - optimal_max) / (max_duration - optimal_max))


# Configurações de detecção de tipos de momento
MOMENT_TYPE_CONFIGS = {
    "revelation": {
        "keywords_pt": ["revelou", "confessou", "admitiu", "primeira vez", "nunca contei"],
        "keywords_en": ["revealed", "confessed", "admitted", "first time", "never told"],
        "weight_boost": 1.3
    },
    "conflict": {
        "keywords_pt": ["discordou", "brigou", "conflito", "discussão", "desentendimento"],
        "keywords_en": ["disagreed", "fought", "conflict", "argument", "dispute"],
        "weight_boost": 1.25
    },
    "humor": {
        "keywords_pt": ["engraçado", "hilário", "piada", "risada", "comédia"],
        "keywords_en": ["funny", "hilarious", "joke", "laughter", "comedy"],
        "weight_boost": 1.2
    },
    "surprise": {
        "keywords_pt": ["surpresa", "inesperado", "chocante", "nossa", "inacreditável"],
        "keywords_en": ["surprise", "unexpected", "shocking", "wow", "unbelievable"],
        "weight_boost": 1.4
    }
}


def get_moment_type_boost(text: str, language: str) -> float:
    """
    Calcula boost baseado no tipo de momento detectado.
    
    Args:
        text: Texto do segmento
        language: Código do idioma
        
    Returns:
        Multiplicador de boost (1.0 = sem boost)
    """
    text_lower = text.lower()
    max_boost = 1.0
    
    for moment_type, config in MOMENT_TYPE_CONFIGS.items():
        keywords_key = f"keywords_{language}"
        if keywords_key in config:
            keywords = config[keywords_key]
            if any(keyword in text_lower for keyword in keywords):
                max_boost = max(max_boost, config["weight_boost"])
    
    return max_boost
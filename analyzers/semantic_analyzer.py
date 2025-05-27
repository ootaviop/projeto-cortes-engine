"""
Analisador semântico multilíngue para detecção de momentos virais.

Este módulo realiza análise semântica profunda utilizando modelos específicos por idioma
para detectar momentos com alto potencial viral através de múltiplas dimensões semânticas.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter

from utils.ml_utils import (
    detect_language, get_model_for_language_and_task, 
    analyze_sentiment, analyze_emotions, get_text_embeddings
)
from config.language_models import get_cultural_config
from config.scoring_weights import get_moment_type_boost
from formatters.json_formatter import TimestampSegment, SemanticAnalysis


@dataclass
class SemanticMetrics:
    """Métricas semânticas calculadas para um segmento."""
    sentiment_score: float
    emotion_intensity: float
    semantic_novelty: float
    information_density: float
    surprise_potential: float
    cultural_relevance: float
    engagement_indicators: Dict[str, float]


class SemanticMomentAnalyzer:
    """Analisador semântico principal para momentos virais."""
    
    def __init__(self):
        """Inicializa o analisador semântico."""
        self.embeddings_cache = {}
        self.analysis_cache = {}
        self.cultural_configs = {}
        
        # Padrões para detecção de características especiais
        self.intensity_patterns = {
            'pt': [
                r'\b(muito|demais|absurdo|inacreditável|impressionante)\b',
                r'\b(nossa|caramba|mano|cara)\b',
                r'[!]{2,}',  # Múltiplas exclamações
                r'\b(pra caramba|pra caralho|muito mesmo)\b'
            ],
            'en': [
                r'\b(very|extremely|absolutely|incredibly|amazing)\b',
                r'\b(wow|omg|dude|man)\b',
                r'[!]{2,}',
                r'\b(so much|really really|absolutely)\b'
            ]
        }
        
        self.surprise_patterns = {
            'pt': [
                r'\b(surpresa|inesperado|chocante|nunca|primeira vez)\b',
                r'\b(não acredito|impossível|incrível)\b',
                r'\b(revelou|confessou|admitiu|contou)\b'
            ],
            'en': [
                r'\b(surprise|unexpected|shocking|never|first time)\b',
                r'\b(can\'t believe|impossible|incredible)\b',
                r'\b(revealed|confessed|admitted|told)\b'
            ]
        }
    
    def analyze_segments(self, segments: List[TimestampSegment], 
                        context_window: int = 3) -> List[Dict[str, Any]]:
        """
        Analisa lista de segmentos com contexto.
        
        Args:
            segments: Lista de segmentos para análise
            context_window: Janela de contexto para análise semântica
            
        Returns:
            Lista de análises semânticas
        """
        if not segments:
            return []
        
        print(f"🧠 Iniciando análise semântica de {len(segments)} segmentos...")
        start_time = time.time()
        
        # Detecta idioma predominante
        all_text = " ".join(seg.text for seg in segments[:5])  # Amostra dos primeiros segmentos
        primary_language = detect_language(all_text)
        print(f"🌐 Idioma detectado: {primary_language}")
        
        # Carrega configuração cultural
        cultural_config = get_cultural_config(primary_language)
        
        # Pré-processa todos os textos
        texts = [seg.text for seg in segments]
        
        # Análise em lote para eficiência
        print("🔄 Analisando sentimentos...")
        sentiment_results = analyze_sentiment(texts, primary_language)
        
        print("🔄 Analisando emoções...")
        emotion_results = analyze_emotions(texts, primary_language)
        
        print("🔄 Gerando embeddings semânticos...")
        embeddings = get_text_embeddings(texts, primary_language)
        
        # Análise individual com contexto
        analyses = []
        for i, segment in enumerate(segments):
            try:
                # Coleta contexto ao redor do segmento
                context_segments = self._get_context_segments(segments, i, context_window)
                
                # Análise semântica completa
                analysis = self._analyze_single_segment(
                    segment, 
                    context_segments,
                    sentiment_results[i],
                    emotion_results[i],
                    embeddings[i],
                    cultural_config,
                    primary_language,
                    embeddings  # Para cálculo de novidade semântica
                )
                
                analyses.append(analysis)
                
            except Exception as e:
                print(f"⚠️ Erro na análise do segmento {i}: {e}")
                # Análise básica como fallback
                analyses.append(self._create_fallback_analysis(segment, sentiment_results[i]))
        
        processing_time = time.time() - start_time
        print(f"✅ Análise semântica concluída ({processing_time:.2f}s)")
        print(f"📊 Média de {len(segments)/processing_time:.1f} segmentos/segundo")
        
        return analyses
    
    def _get_context_segments(self, segments: List[TimestampSegment], 
                            current_idx: int, window_size: int) -> List[TimestampSegment]:
        """Obtém segmentos de contexto ao redor do índice atual."""
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(segments), current_idx + window_size + 1)
        return segments[start_idx:end_idx]
    
    def _analyze_single_segment(self, segment: TimestampSegment,
                              context_segments: List[TimestampSegment],
                              sentiment: Dict[str, Any],
                              emotion: Dict[str, Any],
                              embeddings: List[float],
                              cultural_config: Dict[str, Any],
                              language: str,
                              all_embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Análise semântica completa de um segmento individual.
        
        Args:
            segment: Segmento para análise
            context_segments: Segmentos de contexto
            sentiment: Resultado da análise de sentimento
            emotion: Resultado da análise de emoção
            embeddings: Embeddings do segmento
            cultural_config: Configuração cultural
            language: Código do idioma
            all_embeddings: Embeddings de todos os segmentos
            
        Returns:
            Análise semântica completa
        """
        # Métricas básicas
        metrics = self._calculate_semantic_metrics(
            segment, context_segments, sentiment, emotion, 
            embeddings, cultural_config, language, all_embeddings
        )
        
        # Detecção de tipo de momento
        moment_type = self._detect_moment_type(segment.text, language)
        
        # Indicadores de engajamento
        engagement_indicators = self._analyze_engagement_potential(segment.text, language, cultural_config)
        
        # Análise de contexto narrativo
        narrative_context = self._analyze_narrative_context(segment, context_segments)
        
        # Cria análise semântica formatada
        semantic_analysis = SemanticAnalysis(
            sentiment=sentiment,
            emotion=emotion,
            embeddings=embeddings,
            semantic_novelty=metrics.semantic_novelty,
            information_density=metrics.information_density,
            cultural_context={
                'language': language,
                'cultural_relevance': metrics.cultural_relevance,
                'intensity_boost': cultural_config.get('emotion_multiplier', 1.0),
                'cultural_markers': self._detect_cultural_markers(segment.text, language)
            }
        )
        
        return {
            'segment': segment,
            'semantic_analysis': semantic_analysis,
            'metrics': metrics,
            'moment_type': moment_type,
            'engagement_indicators': engagement_indicators,
            'narrative_context': narrative_context,
            'language': language
        }
    
    def _calculate_semantic_metrics(self, segment: TimestampSegment,
                                  context_segments: List[TimestampSegment],
                                  sentiment: Dict[str, Any],
                                  emotion: Dict[str, Any],
                                  embeddings: List[float],
                                  cultural_config: Dict[str, Any],
                                  language: str,
                                  all_embeddings: List[List[float]]) -> SemanticMetrics:
        """Calcula métricas semânticas para o segmento."""
        
        # Score de sentimento (normalizado)
        sentiment_score = abs(sentiment.get('score', 0.0) - 0.5) * 2  # Converte para 0-1 (extremos são mais virais)
        
        # Intensidade emocional
        emotion_intensity = emotion.get('confidence', 0.0)
        
        # Aplica multiplicador cultural
        emotion_multiplier = cultural_config.get('emotion_multiplier', 1.0)
        emotion_intensity *= emotion_multiplier
        
        # Novidade semântica (distância dos embeddings ao contexto)
        semantic_novelty = self._calculate_semantic_novelty(embeddings, all_embeddings)
        
        # Densidade informacional
        information_density = self._calculate_information_density(segment.text)
        
        # Potencial de surpresa
        surprise_potential = self._calculate_surprise_potential(segment.text, language)
        
        # Relevância cultural
        cultural_relevance = self._calculate_cultural_relevance(segment.text, language, cultural_config)
        
        # Indicadores de engajamento
        engagement_indicators = self._analyze_engagement_potential(segment.text, language, cultural_config)
        
        return SemanticMetrics(
            sentiment_score=min(1.0, sentiment_score),
            emotion_intensity=min(1.0, emotion_intensity),
            semantic_novelty=semantic_novelty,
            information_density=information_density,
            surprise_potential=surprise_potential,
            cultural_relevance=cultural_relevance,
            engagement_indicators=engagement_indicators
        )
    
    def _calculate_semantic_novelty(self, current_embeddings: List[float], 
                                  all_embeddings: List[List[float]]) -> float:
        """Calcula novidade semântica baseada na distância dos embeddings."""
        if len(all_embeddings) < 2:
            return 0.5  # Valor neutro se não há comparação
        
        current_vec = np.array(current_embeddings)
        
        # Calcula distância média para todos os outros embeddings
        distances = []
        for other_embeddings in all_embeddings:
            if other_embeddings != current_embeddings:
                other_vec = np.array(other_embeddings)
                # Distância coseno
                cos_sim = np.dot(current_vec, other_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(other_vec))
                distance = 1 - cos_sim  # Converte similaridade em distância
                distances.append(distance)
        
        if not distances:
            return 0.5
        
        # Normaliza para 0-1
        avg_distance = np.mean(distances)
        return min(1.0, max(0.0, avg_distance))
    
    def _calculate_information_density(self, text: str) -> float:
        """Calcula densidade informacional do texto."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Métricas de densidade
        unique_words = len(set(words))
        total_words = len(words)
        
        # Palavras funcionais (baixa densidade informacional)
        stop_words_pt = {'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'que', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'ela', 'seu', 'sua', 'ou', 'ser', 'ter', 'já', 'foi', 'muito', 'bem', 'pode', 'isso', 'sim', 'só', 'tem', 'vai', 'são', 'está', 'entre', 'sem', 'até', 'pelos'}
        stop_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Detecta idioma e usa stop words apropriadas
        stop_words = stop_words_pt if any(word in stop_words_pt for word in words[:5]) else stop_words_en
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        
        # Diversidade lexical
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Combina métricas
        density_score = (content_ratio * 0.6) + (lexical_diversity * 0.4)
        return min(1.0, density_score)
    
    def _calculate_surprise_potential(self, text: str, language: str) -> float:
        """Calcula potencial de surpresa baseado em padrões linguísticos."""
        text_lower = text.lower()
        surprise_score = 0.0
        
        # Padrões de surpresa por idioma
        patterns = self.surprise_patterns.get(language, self.surprise_patterns.get('en', []))
        
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            surprise_score += matches * 0.2
        
        # Indicadores adicionais
        # Repetição de pontuação
        if '!!' in text or '???' in text:
            surprise_score += 0.3
        
        # Palavras em maiúsculas (indicam ênfase)
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 1])
        surprise_score += caps_words * 0.1
        
        return min(1.0, surprise_score)
    
    def _calculate_cultural_relevance(self, text: str, language: str, 
                                    cultural_config: Dict[str, Any]) -> float:
        """Calcula relevância cultural do conteúdo."""
        text_lower = text.lower()
        relevance_score = 0.0
        
        # Marcadores culturais específicos
        cultural_markers = cultural_config.get('excitement_markers', [])
        intensifiers = cultural_config.get('intensifiers', [])
        
        for marker in cultural_markers:
            if marker.lower() in text_lower:
                relevance_score += 0.2
        
        for intensifier in intensifiers:
            if intensifier.lower() in text_lower:
                relevance_score += 0.15
        
        return min(1.0, relevance_score)
    
    def _detect_moment_type(self, text: str, language: str) -> str:
        """Detecta tipo de momento baseado no conteúdo."""
        text_lower = text.lower()
        
        # Padrões por tipo de momento
        moment_patterns = {
            'revelation': {
                'pt': ['revelou', 'confessou', 'admitiu', 'primeira vez', 'nunca contei', 'segredo'],
                'en': ['revealed', 'confessed', 'admitted', 'first time', 'never told', 'secret']
            },
            'conflict': {
                'pt': ['discordou', 'brigou', 'conflito', 'discussão', 'desentendimento', 'problema'],
                'en': ['disagreed', 'fought', 'conflict', 'argument', 'dispute', 'problem']
            },
            'humor': {
                'pt': ['engraçado', 'hilário', 'piada', 'risada', 'comédia', 'rir'],
                'en': ['funny', 'hilarious', 'joke', 'laughter', 'comedy', 'laugh']
            },
            'surprise': {
                'pt': ['surpresa', 'inesperado', 'chocante', 'inacreditável', 'nossa'],
                'en': ['surprise', 'unexpected', 'shocking', 'unbelievable', 'wow']
            }
        }
        
        # Pontuação por tipo
        type_scores = {}
        for moment_type, patterns in moment_patterns.items():
            score = 0
            lang_patterns = patterns.get(language, patterns.get('en', []))
            
            for pattern in lang_patterns:
                if pattern in text_lower:
                    score += 1
            
            if score > 0:
                type_scores[moment_type] = score
        
        # Retorna tipo com maior pontuação
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _analyze_engagement_potential(self, text: str, language: str, 
                                    cultural_config: Dict[str, Any]) -> Dict[str, float]:
        """Analisa potencial de engajamento do conteúdo."""
        text_lower = text.lower()
        
        indicators = {
            'intensity': 0.0,
            'emotional_appeal': 0.0,
            'shareability': 0.0,
            'controversy_potential': 0.0
        }
        
        # Intensidade
        intensity_patterns = self.intensity_patterns.get(language, self.intensity_patterns.get('en', []))
        for pattern in intensity_patterns:
            matches = len(re.findall(pattern, text_lower))
            indicators['intensity'] += matches * 0.2
        
        # Apelo emocional (baseado em palavras emocionais)
        emotional_words_pt = ['amor', 'ódio', 'medo', 'alegria', 'tristeza', 'raiva', 'surpresa']
        emotional_words_en = ['love', 'hate', 'fear', 'joy', 'sadness', 'anger', 'surprise']
        
        emotional_words = emotional_words_pt if language == 'pt' else emotional_words_en
        for word in emotional_words:
            if word in text_lower:
                indicators['emotional_appeal'] += 0.15
        
        # Shareabilidade (perguntas, chamadas para ação)
        if '?' in text or any(word in text_lower for word in ['você', 'vocês', 'you', 'what do you think']):
            indicators['shareability'] += 0.3
        
        # Potencial de controvérsia
        controversial_markers = ['polêmica', 'controversy', 'discordo', 'disagree', 'errado', 'wrong']
        for marker in controversial_markers:
            if marker in text_lower:
                indicators['controversy_potential'] += 0.2
        
        # Normaliza valores
        for key in indicators:
            indicators[key] = min(1.0, indicators[key])
        
        return indicators
    
    def _analyze_narrative_context(self, segment: TimestampSegment,
                                 context_segments: List[TimestampSegment]) -> Dict[str, Any]:
        """Analisa contexto narrativo do segmento."""
        # Posição na narrativa
        total_segments = len(context_segments)
        segment_position = context_segments.index(segment) if segment in context_segments else 0
        
        narrative_position = "middle"
        if segment_position < total_segments * 0.3:
            narrative_position = "beginning"
        elif segment_position > total_segments * 0.7:
            narrative_position = "end"
        
        # Continuidade com segmentos anteriores/posteriores
        continuity_score = self._calculate_narrative_continuity(segment, context_segments)
        
        return {
            'position': narrative_position,
            'continuity_score': continuity_score,
            'is_climax': continuity_score > 0.8 and narrative_position == "middle",
            'is_resolution': narrative_position == "end" and continuity_score > 0.6
        }
    
    def _calculate_narrative_continuity(self, segment: TimestampSegment,
                                      context_segments: List[TimestampSegment]) -> float:
        """Calcula continuidade narrativa baseada em conectores e temas."""
        if len(context_segments) < 2:
            return 0.5
        
        # Conectores que indicam continuidade
        connectors_pt = ['então', 'aí', 'depois', 'mas', 'porém', 'entretanto', 'assim']
        connectors_en = ['then', 'so', 'after', 'but', 'however', 'therefore', 'thus']
        
        text_lower = segment.text.lower()
        
        # Verifica presença de conectores
        connectors = connectors_pt + connectors_en  # Usa ambos para ser seguro
        connector_score = sum(1 for conn in connectors if conn in text_lower) * 0.2
        
        return min(1.0, connector_score)
    
    def _detect_cultural_markers(self, text: str, language: str) -> List[str]:
        """Detecta marcadores culturais específicos."""
        markers = []
        text_lower = text.lower()
        
        cultural_markers = {
            'pt': {
                'expressões': ['né', 'tá ligado', 'massa', 'top', 'dahora', 'maneiro'],
                'regionalismos': ['mano', 'cara', 'véi', 'galera', 'pessoal'],
                'intensificadores': ['muito', 'demais', 'pra caramba', 'absurdo']
            },
            'en': {
                'expressions': ['you know', 'like', 'awesome', 'cool', 'amazing'],
                'regionalisms': ['dude', 'man', 'folks', 'guys', 'people'],
                'intensifiers': ['very', 'extremely', 'super', 'totally']
            }
        }
        
        lang_markers = cultural_markers.get(language, {})
        for category, marker_list in lang_markers.items():
            for marker in marker_list:
                if marker in text_lower:
                    markers.append(f"{category}:{marker}")
        
        return markers
    
    def _create_fallback_analysis(self, segment: TimestampSegment, 
                                sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Cria análise básica em caso de erro."""
        return {
            'segment': segment,
            'semantic_analysis': SemanticAnalysis(
                sentiment=sentiment,
                emotion={'primary_emotion': 'neutral', 'confidence': 0.5},
                embeddings=None,
                semantic_novelty=0.5,
                information_density=0.5,
                cultural_context={'language': 'unknown'}
            ),
            'metrics': SemanticMetrics(
                sentiment_score=0.5,
                emotion_intensity=0.5,
                semantic_novelty=0.5,
                information_density=0.5,
                surprise_potential=0.5,
                cultural_relevance=0.5,
                engagement_indicators={}
            ),
            'moment_type': 'general',
            'engagement_indicators': {},
            'narrative_context': {'position': 'middle', 'continuity_score': 0.5},
            'language': 'unknown'
        }
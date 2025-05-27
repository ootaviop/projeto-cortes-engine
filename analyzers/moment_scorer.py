"""
Sistema de scoring multi-dimensional para rankeamento de momentos virais.

Este módulo combina métricas semânticas em scores ponderados para identificar
e ranquear momentos com maior potencial viral, considerando diferentes tipos
de conteúdo e contextos culturais.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from config.scoring_weights import (
    get_weights_for_content_type, get_cultural_adjustments, 
    get_duration_config, calculate_duration_score, get_moment_type_boost,
    QUALITY_THRESHOLDS
)
from formatters.json_formatter import TimestampSegment, ViralMoment
from analyzers.semantic_analyzer import SemanticMetrics


@dataclass
class ScoringResult:
    """Resultado do scoring de um momento."""
    viral_score: float
    dimensional_scores: Dict[str, float]
    quality_metrics: Dict[str, float]
    ranking_factors: Dict[str, float]
    recommendations: Dict[str, Any]


class ViralMomentScorer:
    """Sistema de scoring para momentos virais."""
    
    def __init__(self, content_type: str = "default"):
        """
        Inicializa o scorer.
        
        Args:
            content_type: Tipo de conteúdo (podcast, vlog, educational, debate, default)
        """
        self.content_type = content_type
        self.weights = get_weights_for_content_type(content_type)
        self.duration_config = get_duration_config(content_type)
        self.scored_moments = []
    
    def score_moments(self, analyzed_segments: List[Dict[str, Any]], 
                     language: str = "pt") -> List[ViralMoment]:
        """
        Pontua e ranqueia momentos analisados.
        
        Args:
            analyzed_segments: Segmentos com análise semântica
            language: Código do idioma
            
        Returns:
            Lista de momentos virais ranqueados
        """
        if not analyzed_segments:
            return []
        
        print(f"🎯 Iniciando scoring de {len(analyzed_segments)} momentos...")
        start_time = time.time()
        
        # Carrega ajustes culturais
        cultural_adjustments = get_cultural_adjustments(language)
        
        # Calcula scores para todos os segmentos
        scored_segments = []
        for segment_data in analyzed_segments:
            try:
                scoring_result = self._score_single_moment(
                    segment_data, cultural_adjustments, language
                )
                
                if scoring_result.viral_score >= QUALITY_THRESHOLDS["minimum_score"]:
                    scored_segments.append((segment_data, scoring_result))
                    
            except Exception as e:
                print(f"⚠️ Erro no scoring de segmento: {e}")
                continue
        
        # Filtra por qualidade
        quality_moments = self._apply_quality_filters(scored_segments)
        
        # Resolve sobreposições
        optimized_moments = self._resolve_overlaps(quality_moments)
        
        # Ranqueia momentos finais
        ranked_moments = self._rank_moments(optimized_moments)
        
        # Converte para ViralMoment objects
        viral_moments = self._create_viral_moments(ranked_moments)
        
        processing_time = time.time() - start_time
        print(f"✅ Scoring concluído ({processing_time:.2f}s)")
        print(f"📊 {len(viral_moments)} momentos virais identificados")
        
        return viral_moments
    
    def _score_single_moment(self, segment_data: Dict[str, Any], 
                           cultural_adjustments: Dict[str, float],
                           language: str) -> ScoringResult:
        """
        Calcula score de um momento individual.
        
        Args:
            segment_data: Dados do segmento analisado
            cultural_adjustments: Ajustes culturais
            language: Código do idioma
            
        Returns:
            Resultado do scoring
        """
        segment = segment_data['segment']
        metrics = segment_data['metrics']
        moment_type = segment_data['moment_type']
        engagement = segment_data['engagement_indicators']
        
        # Scores dimensionais base
        dimensional_scores = {
            'emotional_intensity': metrics.emotion_intensity,
            'semantic_novelty': metrics.semantic_novelty,
            'surprise_potential': metrics.surprise_potential,
            'information_density': metrics.information_density,
            'narrative_completeness': self._calculate_narrative_completeness(segment_data),
            'engagement_potential': self._calculate_engagement_score(engagement)
        }
        
        # Aplica pesos configurados
        weighted_score = (
            dimensional_scores['emotional_intensity'] * self.weights.emotional_intensity +
            dimensional_scores['semantic_novelty'] * self.weights.semantic_novelty +
            dimensional_scores['surprise_potential'] * self.weights.surprise_potential +
            dimensional_scores['information_density'] * self.weights.information_density +
            dimensional_scores['narrative_completeness'] * self.weights.narrative_completeness +
            dimensional_scores['engagement_potential'] * self.weights.engagement_potential
        )
        
        # Aplica ajustes culturais
        cultural_boost = self._apply_cultural_adjustments(
            dimensional_scores, cultural_adjustments, segment.text
        )
        
        # Boost por tipo de momento
        moment_boost = get_moment_type_boost(segment.text, language)
        
        # Score de duração
        duration_score = calculate_duration_score(segment.duration, self.content_type)
        
        # Score final combinado
        viral_score = weighted_score * cultural_boost * moment_boost
        
        # Penalização por duração inadequada
        if duration_score < 0.5:
            viral_score *= duration_score
        
        # Métricas de qualidade
        quality_metrics = self._calculate_quality_metrics(segment, metrics)
        
        # Fatores de rankeamento
        ranking_factors = {
            'base_score': weighted_score,
            'cultural_boost': cultural_boost,
            'moment_boost': moment_boost,
            'duration_score': duration_score,
            'quality_score': np.mean(list(quality_metrics.values()))
        }
        
        # Recomendações
        recommendations = self._generate_recommendations(
            segment, dimensional_scores, viral_score
        )
        
        return ScoringResult(
            viral_score=min(1.0, max(0.0, viral_score)),
            dimensional_scores=dimensional_scores,
            quality_metrics=quality_metrics,
            ranking_factors=ranking_factors,
            recommendations=recommendations
        )
    
    def _calculate_narrative_completeness(self, segment_data: Dict[str, Any]) -> float:
        """Calcula completude narrativa do segmento."""
        narrative_context = segment_data.get('narrative_context', {})
        
        # Fatores de completude
        completeness_score = 0.0
        
        # Posição na narrativa
        position = narrative_context.get('position', 'middle')
        if position == 'beginning':
            completeness_score += 0.3  # Introduções são importantes
        elif position == 'middle':
            completeness_score += 0.5  # Meio da narrativa
        elif position == 'end':
            completeness_score += 0.4  # Conclusões são relevantes
        
        # Continuidade narrativa
        continuity = narrative_context.get('continuity_score', 0.5)
        completeness_score += continuity * 0.4
        
        # Climax ou resolução
        if narrative_context.get('is_climax', False):
            completeness_score += 0.3
        if narrative_context.get('is_resolution', False):
            completeness_score += 0.2
        
        return min(1.0, completeness_score)
    
    def _calculate_engagement_score(self, engagement_indicators: Dict[str, float]) -> float:
        """Calcula score de potencial de engajamento."""
        if not engagement_indicators:
            return 0.5
        
        # Pesos para diferentes indicadores
        indicator_weights = {
            'intensity': 0.3,
            'emotional_appeal': 0.3,
            'shareability': 0.25,
            'controversy_potential': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for indicator, value in engagement_indicators.items():
            weight = indicator_weights.get(indicator, 0.1)
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _apply_cultural_adjustments(self, dimensional_scores: Dict[str, float],
                                  cultural_adjustments: Dict[str, float],
                                  text: str) -> float:
        """Aplica ajustes culturais aos scores."""
        boost = 1.0
        
        # Boost emocional
        if dimensional_scores['emotional_intensity'] > 0.7:
            boost *= cultural_adjustments.get('emotion_boost', 1.0)
        
        # Boost de surpresa
        if dimensional_scores['surprise_potential'] > 0.6:
            boost *= cultural_adjustments.get('surprise_sensitivity', 1.0)
        
        # Boost de humor (detecção simples)
        humor_markers = ['risos', 'engraçado', 'hilário', 'funny', 'hilarious', 'lol']
        if any(marker in text.lower() for marker in humor_markers):
            boost *= cultural_adjustments.get('humor_detection_boost', 1.0)
        
        # Penalização por sarcasmo
        sarcasm_markers = ['claro', 'obviamente', 'né', 'sure', 'obviously', 'right']
        if any(marker in text.lower() for marker in sarcasm_markers):
            boost *= cultural_adjustments.get('sarcasm_penalty', 1.0)
        
        return boost
    
    def _calculate_quality_metrics(self, segment: TimestampSegment, 
                                 metrics: SemanticMetrics) -> Dict[str, float]:
        """Calcula métricas de qualidade do segmento."""
        quality_metrics = {}
        
        # Qualidade de duração
        duration = segment.duration
        min_dur = QUALITY_THRESHOLDS["minimum_duration"]
        max_dur = QUALITY_THRESHOLDS["maximum_duration"]
        optimal_min = QUALITY_THRESHOLDS["optimal_duration_min"]
        optimal_max = QUALITY_THRESHOLDS["optimal_duration_max"]
        
        if optimal_min <= duration <= optimal_max:
            quality_metrics['duration_quality'] = 1.0
        elif min_dur <= duration <= max_dur:
            quality_metrics['duration_quality'] = 0.7
        else:
            quality_metrics['duration_quality'] = 0.3
        
        # Qualidade de confiança (da transcrição)
        quality_metrics['transcription_confidence'] = segment.confidence
        
        # Qualidade semântica (média das métricas)
        semantic_quality = (
            metrics.semantic_novelty + 
            metrics.information_density + 
            metrics.emotion_intensity
        ) / 3
        quality_metrics['semantic_quality'] = semantic_quality
        
        # Qualidade cultural
        quality_metrics['cultural_relevance'] = metrics.cultural_relevance
        
        return quality_metrics
    
    def _generate_recommendations(self, segment: TimestampSegment,
                                dimensional_scores: Dict[str, float],
                                viral_score: float) -> Dict[str, Any]:
        """Gera recomendações para o momento."""
        recommendations = {
            'cut_priority': 'medium',
            'platforms': [],
            'editing_suggestions': [],
            'content_warnings': []
        }
        
        # Prioridade baseada no score
        if viral_score >= 0.8:
            recommendations['cut_priority'] = 'high'
        elif viral_score >= 0.6:
            recommendations['cut_priority'] = 'medium'
        else:
            recommendations['cut_priority'] = 'low'
        
        # Plataformas recomendadas
        duration = segment.duration
        if 15 <= duration <= 60:
            recommendations['platforms'].extend(['tiktok', 'instagram_reels', 'youtube_shorts'])
        if 30 <= duration <= 180:
            recommendations['platforms'].append('twitter')
        if duration >= 60:
            recommendations['platforms'].append('youtube')
        
        # Sugestões de edição
        if dimensional_scores['emotional_intensity'] > 0.8:
            recommendations['editing_suggestions'].append('highlight_emotional_peaks')
        
        if dimensional_scores['surprise_potential'] > 0.7:
            recommendations['editing_suggestions'].append('add_suspense_buildup')
        
        if dimensional_scores['information_density'] > 0.8:
            recommendations['editing_suggestions'].append('add_visual_aids')
        
        # Avisos de conteúdo
        if dimensional_scores.get('controversy_potential', 0) > 0.7:
            recommendations['content_warnings'].append('potentially_controversial')
        
        return recommendations
    
    def _apply_quality_filters(self, scored_segments: List[Tuple]) -> List[Tuple]:
        """Aplica filtros de qualidade aos segmentos pontuados."""
        filtered_segments = []
        
        for segment_data, scoring_result in scored_segments:
            segment = segment_data['segment']
            
            # Filtro de duração
            if (segment.duration < QUALITY_THRESHOLDS["minimum_duration"] or 
                segment.duration > QUALITY_THRESHOLDS["maximum_duration"]):
                continue
            
            # Filtro de confiança da transcrição
            if segment.confidence < 0.5:
                continue
            
            # Filtro de score mínimo
            if scoring_result.viral_score < QUALITY_THRESHOLDS["minimum_score"]:
                continue
            
            filtered_segments.append((segment_data, scoring_result))
        
        return filtered_segments
    
    def _resolve_overlaps(self, quality_moments: List[Tuple]) -> List[Tuple]:
        """Resolve sobreposições entre momentos, mantendo os melhores."""
        if len(quality_moments) <= 1:
            return quality_moments
        
        # Ordena por score (melhor primeiro)
        sorted_moments = sorted(quality_moments, 
                              key=lambda x: x[1].viral_score, reverse=True)
        
        resolved_moments = []
        
        for current_moment in sorted_moments:
            current_segment = current_moment[0]['segment']
            has_overlap = False
            
            for existing_moment in resolved_moments:
                existing_segment = existing_moment[0]['segment']
                
                # Calcula sobreposição
                overlap = self._calculate_overlap(current_segment, existing_segment)
                
                if overlap > QUALITY_THRESHOLDS["overlap_threshold"]:
                    has_overlap = True
                    break
            
            if not has_overlap:
                resolved_moments.append(current_moment)
        
        return resolved_moments
    
    def _calculate_overlap(self, segment1: TimestampSegment, 
                         segment2: TimestampSegment) -> float:
        """Calcula sobreposição entre dois segmentos."""
        start1, end1 = segment1.start_time, segment1.end_time
        start2, end2 = segment2.start_time, segment2.end_time
        
        # Calcula interseção
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start >= intersection_end:
            return 0.0  # Sem sobreposição
        
        intersection_duration = intersection_end - intersection_start
        
        # Calcula união
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_duration = union_end - union_start
        
        # Sobreposição como porcentagem da união
        return intersection_duration / union_duration if union_duration > 0 else 0.0
    
    def _rank_moments(self, optimized_moments: List[Tuple]) -> List[Tuple]:
        """Ranqueia momentos finais por múltiplos critérios."""
        def ranking_key(moment_tuple):
            segment_data, scoring_result = moment_tuple
            
            # Critérios de rankeamento (ordem de prioridade)
            return (
                scoring_result.viral_score,  # Score viral principal
                scoring_result.ranking_factors['quality_score'],  # Qualidade geral
                -segment_data['segment'].start_time  # Prefere momentos mais cedo (valor negativo)
            )
        
        return sorted(optimized_moments, key=ranking_key, reverse=True)
    
    def _create_viral_moments(self, ranked_moments: List[Tuple]) -> List[ViralMoment]:
        """Converte momentos ranqueados em objetos ViralMoment."""
        viral_moments = []
        
        for rank, (segment_data, scoring_result) in enumerate(ranked_moments):
            segment = segment_data['segment']
            semantic_analysis = segment_data['semantic_analysis']
            moment_type = segment_data['moment_type']
            
            # Cria momento viral
            viral_moment = ViralMoment(
                segment=segment,
                analysis=semantic_analysis,
                viral_score=scoring_result.viral_score,
                emotional_intensity=scoring_result.dimensional_scores['emotional_intensity'],
                surprise_potential=scoring_result.dimensional_scores['surprise_potential'],
                narrative_completeness=scoring_result.dimensional_scores['narrative_completeness'],
                engagement_potential=scoring_result.dimensional_scores['engagement_potential'],
                moment_type=moment_type,
                ranking=rank + 1,
                cut_recommendation=self._create_cut_recommendation(
                    segment, scoring_result, rank + 1
                )
            )
            
            viral_moments.append(viral_moment)
        
        return viral_moments
    
    def _create_cut_recommendation(self, segment: TimestampSegment,
                                 scoring_result: ScoringResult,
                                 ranking: int) -> Dict[str, Any]:
        """Cria recomendação de corte detalhada."""
        recommendations = scoring_result.recommendations
        
        # Calcula timestamps recomendados com padding
        padding_start = min(2.0, segment.start_time)
        padding_end = 1.0
        
        recommended_start = max(0, segment.start_time - padding_start)
        recommended_end = segment.end_time + padding_end
        
        return {
            'recommended_start': round(recommended_start, 2),
            'recommended_end': round(recommended_end, 2),
            'recommended_duration': round(recommended_end - recommended_start, 2),
            'priority': recommendations['cut_priority'],
            'confidence': round(scoring_result.viral_score, 3),
            'ranking': ranking,
            'optimal_for_platform': recommendations['platforms'],
            'editing_suggestions': recommendations['editing_suggestions'],
            'content_warnings': recommendations['content_warnings'],
            'quality_score': round(scoring_result.ranking_factors['quality_score'], 3)
        }
    
    def get_scoring_summary(self, viral_moments: List[ViralMoment]) -> Dict[str, Any]:
        """Gera resumo estatístico do scoring."""
        if not viral_moments:
            return {'error': 'Nenhum momento viral encontrado'}
        
        scores = [moment.viral_score for moment in viral_moments]
        
        return {
            'total_moments': len(viral_moments),
            'high_priority': len([m for m in viral_moments if m.cut_recommendation['priority'] == 'high']),
            'medium_priority': len([m for m in viral_moments if m.cut_recommendation['priority'] == 'medium']),
            'low_priority': len([m for m in viral_moments if m.cut_recommendation['priority'] == 'low']),
            'average_score': round(np.mean(scores), 3),
            'best_score': round(max(scores), 3),
            'score_distribution': {
                'excellent': len([s for s in scores if s >= 0.8]),
                'good': len([s for s in scores if 0.6 <= s < 0.8]),
                'fair': len([s for s in scores if 0.4 <= s < 0.6]),
                'poor': len([s for s in scores if s < 0.4])
            },
            'moment_types': {
                moment_type: len([m for m in viral_moments if m.moment_type == moment_type])
                for moment_type in set(m.moment_type for m in viral_moments)
            }
        }
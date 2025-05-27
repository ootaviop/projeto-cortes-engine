"""
Formatador de dados de análise para output JSON padronizado.

Este módulo estrutura e serializa os resultados da análise semântica em formato
JSON otimizado para consumo pelo módulo VIDEO-CUTTER e outras ferramentas.
"""

import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


@dataclass
class TimestampSegment:
    """Representa um segmento com timestamps."""
    start_time: float
    end_time: float
    duration: float
    text: str
    language: str
    confidence: float


@dataclass
class SemanticAnalysis:
    """Resultado da análise semântica de um segmento."""
    sentiment: Dict[str, Any]
    emotion: Dict[str, Any]
    embeddings: Optional[List[float]]
    semantic_novelty: float
    information_density: float
    cultural_context: Dict[str, Any]


@dataclass
class ViralMoment:
    """Representa um momento com potencial viral."""
    segment: TimestampSegment
    analysis: SemanticAnalysis
    viral_score: float
    emotional_intensity: float
    surprise_potential: float
    narrative_completeness: float
    engagement_potential: float
    moment_type: str
    ranking: int
    cut_recommendation: Dict[str, Any]


@dataclass
class VideoAnalysisResult:
    """Resultado completo da análise de um vídeo."""
    video_info: Dict[str, Any]
    transcription: Dict[str, Any]
    viral_moments: List[ViralMoment]
    analysis_metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]


class AnalysisFormatter:
    """Formatador principal para resultados de análise."""
    
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(self):
        self.creation_time = datetime.now().isoformat()
    
    def create_timestamp_segment(self, start: float, end: float, text: str, 
                                language: str, confidence: float = 1.0) -> TimestampSegment:
        """
        Cria um segmento com timestamps formatado.
        
        Args:
            start: Tempo de início em segundos
            end: Tempo de fim em segundos
            text: Texto transcrito
            language: Código do idioma
            confidence: Confiança da transcrição
            
        Returns:
            TimestampSegment formatado
        """
        return TimestampSegment(
            start_time=round(start, 2),
            end_time=round(end, 2),
            duration=round(end - start, 2),
            text=text.strip(),
            language=language,
            confidence=round(confidence, 3)
        )
    
    def create_semantic_analysis(self, sentiment: Dict, emotion: Dict,
                               embeddings: Optional[List[float]] = None,
                               semantic_novelty: float = 0.0,
                               information_density: float = 0.0,
                               cultural_context: Optional[Dict] = None) -> SemanticAnalysis:
        """
        Cria análise semântica formatada.
        
        Args:
            sentiment: Resultado da análise de sentimento
            emotion: Resultado da análise de emoção
            embeddings: Embeddings semânticos (opcional)
            semantic_novelty: Score de novidade semântica
            information_density: Densidade informacional
            cultural_context: Contexto cultural detectado
            
        Returns:
            SemanticAnalysis formatada
        """
        return SemanticAnalysis(
            sentiment={
                'label': sentiment.get('label', 'neutral'),
                'score': round(sentiment.get('score', 0.0), 3),
                'confidence': round(sentiment.get('confidence', 0.0), 3)
            },
            emotion={
                'primary_emotion': emotion.get('emotion', 'neutral'),
                'confidence': round(emotion.get('confidence', 0.0), 3),
                'all_emotions': emotion.get('all_emotions', {})
            },
            embeddings=embeddings,
            semantic_novelty=round(semantic_novelty, 3),
            information_density=round(information_density, 3),
            cultural_context=cultural_context or {}
        )
    
    def create_viral_moment(self, segment: TimestampSegment, analysis: SemanticAnalysis,
                          scores: Dict[str, float], moment_type: str = "general",
                          ranking: int = 0) -> ViralMoment:
        """
        Cria momento viral formatado.
        
        Args:
            segment: Segmento com timestamps
            analysis: Análise semântica
            scores: Dicionário com scores calculados
            moment_type: Tipo do momento (revelation, conflict, humor, etc.)
            ranking: Posição no ranking
            
        Returns:
            ViralMoment formatado
        """
        # Calcula recomendação de corte
        cut_recommendation = self._generate_cut_recommendation(segment, scores)
        
        return ViralMoment(
            segment=segment,
            analysis=analysis,
            viral_score=round(scores.get('viral_score', 0.0), 3),
            emotional_intensity=round(scores.get('emotional_intensity', 0.0), 3),
            surprise_potential=round(scores.get('surprise_potential', 0.0), 3),
            narrative_completeness=round(scores.get('narrative_completeness', 0.0), 3),
            engagement_potential=round(scores.get('engagement_potential', 0.0), 3),
            moment_type=moment_type,
            ranking=ranking,
            cut_recommendation=cut_recommendation
        )
    
    def _generate_cut_recommendation(self, segment: TimestampSegment, 
                                   scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Gera recomendação de corte para o segmento.
        
        Args:
            segment: Segmento analisado
            scores: Scores calculados
            
        Returns:
            Dicionário com recomendação de corte
        """
        # Ajusta timestamps para corte ótimo
        padding_start = min(2.0, segment.start_time)  # Máximo 2s de padding antes
        padding_end = 1.0  # 1s de padding depois
        
        recommended_start = max(0, segment.start_time - padding_start)
        recommended_end = segment.end_time + padding_end
        recommended_duration = recommended_end - recommended_start
        
        # Determina prioridade baseada no score
        if scores.get('viral_score', 0) >= 0.8:
            priority = "high"
        elif scores.get('viral_score', 0) >= 0.6:
            priority = "medium"
        else:
            priority = "low"
        
        return {
            'recommended_start': round(recommended_start, 2),
            'recommended_end': round(recommended_end, 2),
            'recommended_duration': round(recommended_duration, 2),
            'priority': priority,
            'confidence': round(scores.get('viral_score', 0), 3),
            'optimal_for_platform': self._suggest_platform(recommended_duration),
            'cut_reason': self._generate_cut_reason(scores)
        }
    
    def _suggest_platform(self, duration: float) -> List[str]:
        """Sugere plataformas baseadas na duração."""
        platforms = []
        
        if 15 <= duration <= 60:
            platforms.extend(["tiktok", "instagram_reels", "youtube_shorts"])
        if 30 <= duration <= 180:
            platforms.append("twitter")
        if 60 <= duration <= 600:
            platforms.append("instagram_video")
        if duration >= 60:
            platforms.append("youtube")
        
        return platforms or ["youtube"]
    
    def _generate_cut_reason(self, scores: Dict[str, float]) -> str:
        """Gera explicação para o corte."""
        reasons = []
        
        if scores.get('emotional_intensity', 0) >= 0.7:
            reasons.append("alta intensidade emocional")
        if scores.get('surprise_potential', 0) >= 0.7:
            reasons.append("elemento surpresa")
        if scores.get('narrative_completeness', 0) >= 0.7:
            reasons.append("história completa")
        
        return ", ".join(reasons) or "potencial viral geral"
    
    def create_complete_analysis(self, video_path: str, transcription_data: Dict,
                               viral_moments: List[ViralMoment],
                               processing_stats: Dict[str, Any]) -> VideoAnalysisResult:
        """
        Cria resultado completo da análise.
        
        Args:
            video_path: Caminho do vídeo analisado
            transcription_data: Dados da transcrição
            viral_moments: Lista de momentos virais
            processing_stats: Estatísticas de processamento
            
        Returns:
            VideoAnalysisResult completo
        """
        video_path = Path(video_path)
        
        video_info = {
            'filename': video_path.name,
            'filepath': str(video_path),
            'size_mb': round(video_path.stat().st_size / (1024*1024), 2) if video_path.exists() else 0,
            'duration': transcription_data.get('duration', 0),
            'primary_language': transcription_data.get('language', 'unknown')
        }
        
        analysis_metadata = {
            'schema_version': self.SCHEMA_VERSION,
            'created_at': self.creation_time,
            'total_moments_found': len(viral_moments),
            'high_priority_moments': len([m for m in viral_moments if m.cut_recommendation['priority'] == 'high']),
            'average_viral_score': round(sum(m.viral_score for m in viral_moments) / len(viral_moments), 3) if viral_moments else 0.0,
            'languages_detected': list(set(m.segment.language for m in viral_moments)),
            'moment_types': list(set(m.moment_type for m in viral_moments))
        }
        
        return VideoAnalysisResult(
            video_info=video_info,
            transcription=transcription_data,
            viral_moments=viral_moments,
            analysis_metadata=analysis_metadata,
            processing_stats=processing_stats
        )
    
    def to_json(self, analysis_result: VideoAnalysisResult, indent: int = 2) -> str:
        """
        Converte resultado para JSON string.
        
        Args:
            analysis_result: Resultado da análise
            indent: Indentação do JSON
            
        Returns:
            String JSON formatada
        """
        def serialize_dataclass(obj):
            """Serializa dataclasses para JSON."""
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            elif isinstance(obj, (list, tuple)):
                return [serialize_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: serialize_dataclass(value) for key, value in obj.items()}
            else:
                return obj
        
        serializable_data = serialize_dataclass(analysis_result)
        return json.dumps(serializable_data, indent=indent, ensure_ascii=False)
    
    def save_to_file(self, analysis_result: VideoAnalysisResult, output_path: str) -> str:
        """
        Salva resultado em arquivo JSON.
        
        Args:
            analysis_result: Resultado da análise
            output_path: Caminho do arquivo de saída
            
        Returns:
            Caminho do arquivo salvo
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_content = self.to_json(analysis_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        return str(output_path)
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Carrega análise de arquivo JSON.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Dicionário com dados da análise
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_summary_report(self, analysis_result: VideoAnalysisResult) -> Dict[str, Any]:
        """
        Cria relatório resumido da análise.
        
        Args:
            analysis_result: Resultado da análise
            
        Returns:
            Dicionário com resumo
        """
        high_priority = [m for m in analysis_result.viral_moments if m.cut_recommendation['priority'] == 'high']
        
        return {
            'video_filename': analysis_result.video_info['filename'],
            'total_duration': analysis_result.video_info['duration'],
            'moments_found': len(analysis_result.viral_moments),
            'high_priority_moments': len(high_priority),
            'best_moment': {
                'start': high_priority[0].segment.start_time if high_priority else None,
                'score': high_priority[0].viral_score if high_priority else None,
                'type': high_priority[0].moment_type if high_priority else None
            },
            'recommended_cuts': len([m for m in analysis_result.viral_moments if m.viral_score >= 0.6]),
            'processing_time': analysis_result.processing_stats.get('total_time', 0),
            'success': True
        }


def validate_analysis_json(json_data: Dict[str, Any]) -> List[str]:
    """
    Valida estrutura do JSON de análise.
    
    Args:
        json_data: Dados JSON para validação
        
    Returns:
        Lista de erros encontrados (vazia se válido)
    """
    errors = []
    
    required_fields = ['video_info', 'transcription', 'viral_moments', 'analysis_metadata']
    
    for field in required_fields:
        if field not in json_data:
            errors.append(f"Campo obrigatório ausente: {field}")
    
    if 'viral_moments' in json_data:
        for i, moment in enumerate(json_data['viral_moments']):
            if 'segment' not in moment:
                errors.append(f"Momento {i}: campo 'segment' ausente")
            if 'viral_score' not in moment:
                errors.append(f"Momento {i}: campo 'viral_score' ausente")
    
    return errors
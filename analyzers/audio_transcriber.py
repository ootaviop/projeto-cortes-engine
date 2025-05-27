"""
Transcritor de áudio com segmentação inteligente para análise semântica.

Este módulo utiliza Whisper OpenAI para transcrição multilíngue com timestamps
precisos e segmentação automática baseada em pausas naturais e mudanças de contexto.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import whisper
import torch
from dataclasses import dataclass

from utils.audio_utils import AudioProcessor, check_ffmpeg_availability
from utils.file_manager import FileManager
from formatters.json_formatter import TimestampSegment


@dataclass
class TranscriptionWord:
    """Representa uma palavra com timestamp preciso."""
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionSegment:
    """Representa um segmento de transcrição."""
    id: int
    start: float
    end: float
    text: str
    words: List[TranscriptionWord]
    language: str
    confidence: float
    no_speech_prob: float


class AudioTranscriber:
    """Transcritor de áudio com segmentação inteligente."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Inicializa o transcritor.
        
        Args:
            model_size: Tamanho do modelo Whisper (tiny, base, small, medium, large)
            device: Dispositivo para processamento (auto-detecta se None)
        """
        self.model_size = model_size
        self.device = device or self._get_optimal_device()
        self.model = None
        self.audio_processor = AudioProcessor()
        self.file_manager = FileManager()
        
        # Configurações de transcrição
        self.whisper_options = {
            "language": None,  # Auto-detecta
            "task": "transcribe",
            "verbose": False,
            "word_timestamps": True,
            "temperature": 0.0,  # Determinístico
            "best_of": 1,
            "beam_size": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "suppress_tokens": [-1],
            "initial_prompt": None,
            "condition_on_previous_text": True,
            "fp16": torch.cuda.is_available(),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6
        }
    
    def _get_optimal_device(self) -> str:
        """Determina o dispositivo ótimo para Whisper."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Carrega o modelo Whisper se necessário."""
        if self.model is None:
            print(f"🔄 Carregando Whisper modelo '{self.model_size}' em {self.device}")
            start_time = time.time()
            
            try:
                self.model = whisper.load_model(self.model_size, device=self.device)
                load_time = time.time() - start_time
                print(f"✅ Modelo Whisper carregado ({load_time:.2f}s)")
            except Exception as e:
                print(f"❌ Erro ao carregar Whisper: {e}")
                raise
    
    def transcribe_video(self, video_path: str, force_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcreve vídeo completo com segmentação inteligente.
        
        Args:
            video_path: Caminho do arquivo de vídeo
            force_language: Força idioma específico (opcional)
            
        Returns:
            Dicionário com transcrição completa e metadados
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        # Verifica dependências
        if not check_ffmpeg_availability():
            raise RuntimeError("ffmpeg não encontrado. Instale ffmpeg para continuar.")
        
        print(f"🎙️ Iniciando transcrição: {video_path.name}")
        start_time = time.time()
        
        try:
            # Extrai áudio do vídeo
            print("🔄 Extraindo áudio...")
            audio_path = self._extract_audio(video_path)
            
            # Valida qualidade do áudio
            quality_check = self.audio_processor.validate_audio_quality(audio_path)
            if not quality_check['is_valid']:
                raise RuntimeError(f"Qualidade de áudio inadequada: {quality_check.get('error', 'Unknown')}")
            
            print(f"✅ Áudio extraído (qualidade: {quality_check['quality_score']:.2f})")
            
            # Carrega modelo e transcreve
            self._load_model()
            
            # Configura opções específicas
            options = self.whisper_options.copy()
            if force_language:
                options["language"] = force_language
            
            print("🔄 Transcrevendo áudio...")
            result = self.model.transcribe(audio_path, **options)
            
            # Processa resultado
            transcription_data = self._process_whisper_result(result, video_path)
            
            # Segmentação inteligente
            print("🔄 Aplicando segmentação inteligente...")
            intelligent_segments = self._create_intelligent_segments(transcription_data)
            
            # Calcula estatísticas
            processing_time = time.time() - start_time
            stats = self._calculate_transcription_stats(transcription_data, processing_time)
            
            # Monta resultado final
            final_result = {
                'transcription': transcription_data,
                'intelligent_segments': intelligent_segments,
                'statistics': stats,
                'audio_info': self.audio_processor.get_audio_info(audio_path),
                'processing_time': processing_time
            }
            
            print(f"✅ Transcrição concluída ({processing_time:.2f}s)")
            print(f"📊 {len(intelligent_segments)} segmentos inteligentes criados")
            
            # Limpeza
            self._cleanup_temp_files(audio_path)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Erro na transcrição: {e}")
            # Limpeza em caso de erro
            try:
                self._cleanup_temp_files(audio_path)
            except:
                pass
            raise
    
    def _extract_audio(self, video_path: str) -> str:
        """Extrai áudio do vídeo para transcrição."""
        return self.audio_processor.extract_audio_from_video(video_path)
    
    def _process_whisper_result(self, result: Dict, video_path: Path) -> Dict[str, Any]:
        """
        Processa resultado bruto do Whisper.
        
        Args:
            result: Resultado do Whisper
            video_path: Caminho do vídeo original
            
        Returns:
            Dados de transcrição processados
        """
        segments = []
        
        for seg_data in result.get("segments", []):
            # Processa palavras com timestamps
            words = []
            if "words" in seg_data:
                for word_data in seg_data["words"]:
                    words.append(TranscriptionWord(
                        word=word_data["word"].strip(),
                        start=round(word_data["start"], 2),
                        end=round(word_data["end"], 2),
                        confidence=word_data.get("probability", 1.0)
                    ))
            
            # Cria segmento
            segment = TranscriptionSegment(
                id=seg_data["id"],
                start=round(seg_data["start"], 2),
                end=round(seg_data["end"], 2),
                text=seg_data["text"].strip(),
                words=words,
                language=result.get("language", "unknown"),
                confidence=1.0 - seg_data.get("no_speech_prob", 0.0),
                no_speech_prob=seg_data.get("no_speech_prob", 0.0)
            )
            segments.append(segment)
        
        return {
            'filename': video_path.name,
            'language': result.get("language", "unknown"),
            'duration': max(seg.end for seg in segments) if segments else 0.0,
            'text': result.get("text", "").strip(),
            'segments': segments,
            'word_count': sum(len(seg.words) for seg in segments),
            'confidence': sum(seg.confidence for seg in segments) / len(segments) if segments else 0.0
        }
    
    def _create_intelligent_segments(self, transcription_data: Dict) -> List[TimestampSegment]:
        """
        Cria segmentos inteligentes baseados em pausas e contexto.
        
        Args:
            transcription_data: Dados da transcrição
            
        Returns:
            Lista de segmentos inteligentes
        """
        segments = transcription_data['segments']
        if not segments:
            return []
        
        intelligent_segments = []
        current_segment_data = {
            'start': segments[0].start,
            'text_parts': [],
            'words_count': 0,
            'language': transcription_data['language']
        }
        
        for i, segment in enumerate(segments):
            # Adiciona texto do segmento atual
            current_segment_data['text_parts'].append(segment.text)
            current_segment_data['words_count'] += len(segment.words)
            
            # Verifica se deve finalizar o segmento
            should_split = self._should_split_segment(
                segment, segments[i+1] if i+1 < len(segments) else None, 
                current_segment_data
            )
            
            if should_split or i == len(segments) - 1:
                # Finaliza segmento atual
                final_text = " ".join(current_segment_data['text_parts']).strip()
                
                if final_text:  # Só adiciona se tiver conteúdo
                    intelligent_segment = TimestampSegment(
                        start_time=current_segment_data['start'],
                        end_time=segment.end,
                        duration=segment.end - current_segment_data['start'],
                        text=final_text,
                        language=current_segment_data['language'],
                        confidence=segment.confidence
                    )
                    intelligent_segments.append(intelligent_segment)
                
                # Inicia novo segmento se não for o último
                if i < len(segments) - 1:
                    current_segment_data = {
                        'start': segments[i+1].start,
                        'text_parts': [],
                        'words_count': 0,
                        'language': transcription_data['language']
                    }
        
        return intelligent_segments
    
    def _should_split_segment(self, current_seg: TranscriptionSegment, 
                            next_seg: Optional[TranscriptionSegment],
                            segment_data: Dict) -> bool:
        """
        Determina se deve dividir o segmento atual.
        
        Args:
            current_seg: Segmento atual
            next_seg: Próximo segmento (se existir)
            segment_data: Dados do segmento sendo construído
            
        Returns:
            True se deve dividir
        """
        # Duração máxima por segmento (30 segundos)
        current_duration = current_seg.end - segment_data['start']
        if current_duration >= 30.0:
            return True
        
        # Número máximo de palavras por segmento
        if segment_data['words_count'] >= 50:
            return True
        
        # Pausa longa entre segmentos
        if next_seg and (next_seg.start - current_seg.end) >= 2.0:
            return True
        
        # Detecta fim de frase/parágrafo
        text = current_seg.text.strip()
        if text.endswith(('.', '!', '?', '...')):
            return True
        
        # Detecta mudança de tópico (simplificado)
        if text.lower().startswith(('então', 'mas', 'porém', 'agora', 'depois')):
            return True
        
        return False
    
    def _calculate_transcription_stats(self, transcription_data: Dict, 
                                     processing_time: float) -> Dict[str, Any]:
        """Calcula estatísticas da transcrição."""
        segments = transcription_data['segments']
        
        if not segments:
            return {'error': 'Nenhum segmento encontrado'}
        
        total_words = sum(len(seg.words) for seg in segments)
        avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
        
        return {
            'total_segments': len(segments),
            'total_words': total_words,
            'duration': transcription_data['duration'],
            'language': transcription_data['language'],
            'avg_confidence': round(avg_confidence, 3),
            'words_per_minute': round((total_words / transcription_data['duration']) * 60, 1) if transcription_data['duration'] > 0 else 0,
            'processing_time': round(processing_time, 2),
            'processing_speed': round(transcription_data['duration'] / processing_time, 2) if processing_time > 0 else 0,
            'model_used': self.model_size,
            'device_used': self.device
        }
    
    def _cleanup_temp_files(self, audio_path: str):
        """Remove arquivos temporários."""
        try:
            audio_file = Path(audio_path)
            if audio_file.exists() and "temp" in str(audio_file):
                audio_file.unlink()
            self.audio_processor.cleanup_temp_files()
        except Exception as e:
            print(f"⚠️ Aviso: Erro na limpeza de arquivos temporários: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """Retorna lista de idiomas suportados pelo Whisper."""
        return [
            "pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh", "ru",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ]
    
    def estimate_processing_time(self, video_path: str) -> float:
        """
        Estima tempo de processamento para um vídeo.
        
        Args:
            video_path: Caminho do vídeo
            
        Returns:
            Tempo estimado em segundos
        """
        try:
            # Extrai informações básicas do vídeo
            temp_audio = self._extract_audio(video_path)
            audio_info = self.audio_processor.get_audio_info(temp_audio)
            duration = audio_info['duration']
            
            # Limpeza
            self._cleanup_temp_files(temp_audio)
            
            # Fatores de velocidade por modelo
            speed_factors = {
                "tiny": 20.0,    # 20x mais rápido que tempo real
                "base": 10.0,    # 10x mais rápido
                "small": 5.0,    # 5x mais rápido
                "medium": 2.0,   # 2x mais rápido
                "large": 1.0     # Tempo real
            }
            
            factor = speed_factors.get(self.model_size, 2.0)
            if self.device == "cpu":
                factor *= 0.3  # CPU é ~3x mais lenta
            
            return duration / factor
            
        except Exception:
            # Fallback: estimativa conservadora
            return 60.0  # 1 minuto como padrão
"""
Utilitários para processamento de áudio e extração de características.

Este módulo fornece funcionalidades especializadas para operações de áudio
necessárias para análise semântica, incluindo extração, normalização e
preparação de dados para modelos de ML.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import json


class AudioProcessor:
    """Processador de áudio para operações especializadas."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "cortes_audio_temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extrai áudio de vídeo usando ffmpeg.
        
        Args:
            video_path: Caminho do arquivo de vídeo
            output_path: Caminho de saída (opcional)
            
        Returns:
            Caminho do arquivo de áudio extraído
            
        Raises:
            RuntimeError: Se a extração falhar
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        if output_path is None:
            output_path = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Comando ffmpeg para extração otimizada
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # Sem vídeo
            "-acodec", "pcm_s16le",  # Codec PCM 16-bit
            "-ar", "16000",  # Sample rate 16kHz (otimizado para Whisper)
            "-ac", "1",  # Mono
            "-y",  # Sobrescrever
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Erro na extração de áudio: {e.stderr}")
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        Obtém informações detalhadas do arquivo de áudio.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            Dicionário com informações do áudio
        """
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extrai informações relevantes
            audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
            
            return {
                "duration": float(info["format"]["duration"]),
                "sample_rate": int(audio_stream["sample_rate"]),
                "channels": int(audio_stream["channels"]),
                "bit_rate": int(audio_stream.get("bit_rate", 0)),
                "codec": audio_stream["codec_name"],
                "size_bytes": int(info["format"]["size"])
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Erro ao obter informações do áudio: {e}")
    
    def detect_silence_segments(self, audio_path: str, silence_threshold: float = -30.0, 
                              min_silence_duration: float = 1.0) -> List[Tuple[float, float]]:
        """
        Detecta segmentos de silêncio no áudio.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            silence_threshold: Threshold de silêncio em dB
            min_silence_duration: Duração mínima de silêncio em segundos
            
        Returns:
            Lista de tuplas (início, fim) dos segmentos de silêncio
        """
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", f"silencedetect=noise={silence_threshold}dB:duration={min_silence_duration}",
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            silence_segments = []
            
            # Parseia a saída para encontrar segmentos de silêncio
            lines = result.stderr.split('\n')
            silence_start = None
            
            for line in lines:
                if "silence_start" in line:
                    # Extrai timestamp de início do silêncio
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "silence_start:":
                            silence_start = float(parts[i + 1])
                            break
                
                elif "silence_end" in line and silence_start is not None:
                    # Extrai timestamp de fim do silêncio
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "silence_end:":
                            silence_end = float(parts[i + 1])
                            silence_segments.append((silence_start, silence_end))
                            silence_start = None
                            break
            
            return silence_segments
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Erro na detecção de silêncio: {e}")
    
    def normalize_audio_volume(self, audio_path: str, target_path: str, 
                             target_lufs: float = -23.0) -> str:
        """
        Normaliza o volume do áudio para um nível específico.
        
        Args:
            audio_path: Caminho do áudio original
            target_path: Caminho do áudio normalizado
            target_lufs: Nível LUFS alvo
            
        Returns:
            Caminho do arquivo normalizado
        """
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-y", target_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return target_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Erro na normalização de áudio: {e}")
    
    def split_audio_by_segments(self, audio_path: str, segments: List[Tuple[float, float]], 
                               output_dir: str) -> List[str]:
        """
        Divide áudio em segmentos baseado em timestamps.
        
        Args:
            audio_path: Caminho do áudio original
            segments: Lista de tuplas (início, fim) em segundos
            output_dir: Diretório de saída
            
        Returns:
            Lista de caminhos dos segmentos criados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_name = Path(audio_path).stem
        segment_paths = []
        
        for i, (start, end) in enumerate(segments):
            segment_path = output_dir / f"{audio_name}_segment_{i:03d}.wav"
            
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", str(start),
                "-t", str(end - start),
                "-c", "copy",
                "-y", str(segment_path)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                segment_paths.append(str(segment_path))
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Erro ao criar segmento {i}: {e}")
                continue
        
        return segment_paths
    
    def validate_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """
        Valida a qualidade do áudio para análise.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            Dicionário com métricas de qualidade
        """
        try:
            info = self.get_audio_info(audio_path)
            
            quality_metrics = {
                "is_valid": True,
                "sample_rate_ok": info["sample_rate"] >= 8000,  # Mínimo para speech
                "duration_ok": info["duration"] >= 1.0,  # Mínimo 1 segundo
                "size_ok": info["size_bytes"] > 1000,  # Mínimo 1KB
                "codec_supported": info["codec"] in ["pcm_s16le", "aac", "mp3", "wav"],
                "quality_score": 0.0
            }
            
            # Calcula score de qualidade
            score = 0.0
            if quality_metrics["sample_rate_ok"]:
                score += 0.3
            if quality_metrics["duration_ok"]:
                score += 0.3
            if quality_metrics["size_ok"]:
                score += 0.2
            if quality_metrics["codec_supported"]:
                score += 0.2
            
            quality_metrics["quality_score"] = score
            quality_metrics["is_valid"] = score >= 0.8
            
            return quality_metrics
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "quality_score": 0.0
            }
    
    def cleanup_temp_files(self):
        """Remove arquivos temporários criados."""
        try:
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
        except Exception as e:
            print(f"⚠️ Erro na limpeza de arquivos temporários: {e}")


def check_ffmpeg_availability() -> bool:
    """
    Verifica se o ffmpeg está disponível no sistema.
    
    Returns:
        True se ffmpeg estiver disponível
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_optimal_audio_format() -> Dict[str, str]:
    """
    Retorna configurações ótimas de áudio para análise ML.
    
    Returns:
        Dicionário com parâmetros de áudio
    """
    return {
        "sample_rate": "16000",  # 16kHz - ótimo para Whisper
        "channels": "1",         # Mono - reduz complexidade
        "codec": "pcm_s16le",    # PCM 16-bit - qualidade máxima
        "format": "wav"          # WAV - sem compressão
    }


def estimate_processing_time(audio_duration: float) -> float:
    """
    Estima tempo de processamento baseado na duração do áudio.
    
    Args:
        audio_duration: Duração do áudio em segundos
        
    Returns:
        Tempo estimado de processamento em segundos
    """
    # Baseado em benchmarks empíricos
    # Whisper: ~0.1x tempo real
    # Análise semântica: ~0.05x tempo real
    # Overhead: ~20%
    return audio_duration * 0.15 * 1.2
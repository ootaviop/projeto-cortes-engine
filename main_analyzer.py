"""
Pipeline principal de análise de vídeos para detecção de momentos virais.

Este módulo orquestra todo o processo de análise, desde a transcrição até o scoring,
fornecendo uma interface CLI amigável para operadores não técnicos.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
from tqdm import tqdm

# Importações dos módulos do projeto
from utils.file_manager import FileManager
from utils.audio_utils import check_ffmpeg_availability
from analyzers.audio_transcriber import AudioTranscriber
from analyzers.semantic_analyzer import SemanticMomentAnalyzer
from analyzers.moment_scorer import ViralMomentScorer
from formatters.json_formatter import AnalysisFormatter
from config.language_models import LanguageModels


class VideoAnalysisPipeline:
    """Pipeline completo de análise de vídeos."""
    
    def __init__(self, whisper_model: str = "base", content_type: str = "default"):
        """
        Inicializa o pipeline de análise.
        
        Args:
            whisper_model: Modelo Whisper a ser usado (tiny, base, small, medium, large)
            content_type: Tipo de conteúdo (podcast, vlog, educational, debate, default)
        """
        self.whisper_model = whisper_model
        self.content_type = content_type
        
        # Inicializa componentes
        self.file_manager = FileManager()
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.semantic_analyzer = SemanticMomentAnalyzer()
        self.moment_scorer = ViralMomentScorer(content_type=content_type)
        self.formatter = AnalysisFormatter()
        
        # Configuração de diretórios
        self.input_dir = Path("downloads")
        self.output_dir = Path("analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Pipeline inicializado:")
        print(f"   📁 Input: {self.input_dir}")
        print(f"   📁 Output: {self.output_dir}")
        print(f"   🎙️ Whisper: {whisper_model}")
        print(f"   📊 Tipo: {content_type}")
    
    def analyze_video(self, video_path: str, force_language: Optional[str] = None,
                     min_viral_score: float = 0.6) -> Dict[str, Any]:
        """
        Analisa um vídeo completo.
        
        Args:
            video_path: Caminho do arquivo de vídeo
            force_language: Força idioma específico (opcional)
            min_viral_score: Score mínimo para considerar momento viral
            
        Returns:
            Resultado completo da análise
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        print(f"\n🎬 Analisando: {video_path.name}")
        print("=" * 60)
        
        analysis_start_time = time.time()
        processing_stats = {
            'video_file': str(video_path),
            'start_time': time.time(),
            'stages': {}
        }
        
        try:
            # ETAPA 1: Transcrição
            print("🎙️ ETAPA 1: Transcrição de áudio...")
            stage_start = time.time()
            
            transcription_result = self.transcriber.transcribe_video(
                str(video_path), force_language
            )
            
            stage_time = time.time() - stage_start
            processing_stats['stages']['transcription'] = {
                'duration': stage_time,
                'segments_created': len(transcription_result['intelligent_segments']),
                'confidence': transcription_result['transcription']['confidence']
            }
            
            print(f"✅ Transcrição concluída ({stage_time:.2f}s)")
            print(f"   📝 {len(transcription_result['intelligent_segments'])} segmentos inteligentes")
            print(f"   🎯 Confiança média: {transcription_result['transcription']['confidence']:.2f}")
            
            # ETAPA 2: Análise Semântica
            print("\n🧠 ETAPA 2: Análise semântica...")
            stage_start = time.time()
            
            semantic_analyses = self.semantic_analyzer.analyze_segments(
                transcription_result['intelligent_segments']
            )
            
            stage_time = time.time() - stage_start
            processing_stats['stages']['semantic_analysis'] = {
                'duration': stage_time,
                'segments_analyzed': len(semantic_analyses),
                'primary_language': transcription_result['transcription']['language']
            }
            
            print(f"✅ Análise semântica concluída ({stage_time:.2f}s)")
            print(f"   🌐 Idioma: {transcription_result['transcription']['language']}")
            
            # ETAPA 3: Scoring e Rankeamento
            print("\n🎯 ETAPA 3: Scoring de momentos...")
            stage_start = time.time()
            
            viral_moments = self.moment_scorer.score_moments(
                semantic_analyses, 
                transcription_result['transcription']['language']
            )
            
            # Filtra por score mínimo
            filtered_moments = [
                moment for moment in viral_moments 
                if moment.viral_score >= min_viral_score
            ]
            
            stage_time = time.time() - stage_start
            processing_stats['stages']['scoring'] = {
                'duration': stage_time,
                'total_moments': len(viral_moments),
                'filtered_moments': len(filtered_moments),
                'min_score_threshold': min_viral_score
            }
            
            print(f"✅ Scoring concluído ({stage_time:.2f}s)")
            print(f"   🎯 {len(filtered_moments)} momentos virais (score ≥ {min_viral_score})")
            
            # ETAPA 4: Formatação de Resultados
            print("\n📄 ETAPA 4: Formatação de resultados...")
            stage_start = time.time()
            
            total_analysis_time = time.time() - analysis_start_time
            processing_stats['total_time'] = total_analysis_time
            processing_stats['end_time'] = time.time()
            
            # Cria resultado completo
            complete_result = self.formatter.create_complete_analysis(
                str(video_path),
                transcription_result['transcription'],
                filtered_moments,
                processing_stats
            )
            
            stage_time = time.time() - stage_start
            processing_stats['stages']['formatting'] = {
                'duration': stage_time
            }
            
            print(f"✅ Formatação concluída ({stage_time:.2f}s)")
            
            # Salva resultado
            output_filename = f"{video_path.stem}_analysis.json"
            output_path = self.output_dir / output_filename
            
            saved_path = self.formatter.save_to_file(complete_result, output_path)
            
            # Estatísticas finais
            print(f"\n📊 ANÁLISE CONCLUÍDA!")
            print("=" * 60)
            print(f"⏱️  Tempo total: {total_analysis_time:.2f}s")
            print(f"🎯 Momentos virais: {len(filtered_moments)}")
            print(f"📁 Resultado salvo: {saved_path}")
            
            if filtered_moments:
                best_moment = filtered_moments[0]
                print(f"🏆 Melhor momento:")
                print(f"   ⏰ {best_moment.segment.start_time:.1f}s - {best_moment.segment.end_time:.1f}s")
                print(f"   🎯 Score: {best_moment.viral_score:.3f}")
                print(f"   🏷️  Tipo: {best_moment.moment_type}")
            
            return {
                'success': True,
                'result': complete_result,
                'output_file': saved_path,
                'summary': self.formatter.create_summary_report(complete_result)
            }
            
        except Exception as e:
            error_msg = f"Erro na análise: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Log detalhado do erro
            error_details = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'video_file': str(video_path),
                'processing_stats': processing_stats
            }
            
            return {
                'success': False,
                'error': error_msg,
                'details': error_details
            }
    
    def analyze_batch(self, input_directory: Optional[str] = None,
                     file_pattern: str = "*.mp4",
                     min_viral_score: float = 0.6) -> Dict[str, Any]:
        """
        Analisa múltiplos vídeos em lote.
        
        Args:
            input_directory: Diretório de entrada (usa self.input_dir se None)
            file_pattern: Padrão de arquivos a processar
            min_viral_score: Score mínimo para momentos virais
            
        Returns:
            Resultado do processamento em lote
        """
        input_dir = Path(input_directory) if input_directory else self.input_dir
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
        
        # Encontra arquivos de vídeo
        video_files = list(input_dir.glob(file_pattern))
        
        if not video_files:
            return {
                'success': False,
                'error': f"Nenhum arquivo {file_pattern} encontrado em {input_dir}"
            }
        
        print(f"🎬 Processamento em lote iniciado")
        print(f"📁 Diretório: {input_dir}")
        print(f"🔍 Padrão: {file_pattern}")
        print(f"📹 {len(video_files)} vídeos encontrados")
        print("=" * 60)
        
        batch_results = {
            'success': True,
            'total_videos': len(video_files),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'results': [],
            'errors': [],
            'total_moments_found': 0,
            'processing_time': 0
        }
        
        batch_start_time = time.time()
        
        # Processa cada vídeo com barra de progresso
        for video_file in tqdm(video_files, desc="Analisando vídeos"):
            try:
                result = self.analyze_video(str(video_file), min_viral_score=min_viral_score)
                
                batch_results['processed'] += 1
                
                if result['success']:
                    batch_results['successful'] += 1
                    batch_results['results'].append(result)
                    
                    # Conta momentos encontrados
                    summary = result.get('summary', {})
                    moments = summary.get('moments_found', 0)
                    batch_results['total_moments_found'] += moments
                    
                else:
                    batch_results['failed'] += 1
                    batch_results['errors'].append({
                        'file': str(video_file),
                        'error': result.get('error', 'Erro desconhecido')
                    })
                
            except Exception as e:
                batch_results['failed'] += 1
                batch_results['errors'].append({
                    'file': str(video_file),
                    'error': str(e)
                })
                print(f"❌ Erro ao processar {video_file.name}: {e}")
        
        batch_results['processing_time'] = time.time() - batch_start_time
        
        # Relatório final
        print("\n📊 RELATÓRIO DO LOTE")
        print("=" * 60)
        print(f"✅ Processados: {batch_results['successful']}/{batch_results['total_videos']}")
        print(f"❌ Falhas: {batch_results['failed']}")
        print(f"🎯 Total de momentos: {batch_results['total_moments_found']}")
        print(f"⏱️  Tempo total: {batch_results['processing_time']:.2f}s")
        
        if batch_results['errors']:
            print(f"\n⚠️  ERROS:")
            for error in batch_results['errors']:
                print(f"   📹 {Path(error['file']).name}: {error['error']}")
        
        return batch_results
    
    def validate_dependencies(self) -> bool:
        """Valida dependências necessárias."""
        print("🔍 Validando dependências...")
        
        dependencies_ok = True
        
        # Verifica ffmpeg
        if not check_ffmpeg_availability():
            print("❌ ffmpeg não encontrado")
            print("   Instale com: sudo apt install ffmpeg  # ou equivalente")
            dependencies_ok = False
        else:
            print("✅ ffmpeg disponível")
        
        # Verifica diretórios
        if not self.input_dir.exists():
            print(f"⚠️  Diretório de entrada não existe: {self.input_dir}")
            print("   Criando diretório...")
            self.input_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Diretórios configurados")
        
        # Verifica modelos (carregamento básico)
        try:
            supported_languages = LanguageModels.get_supported_languages()
            print(f"✅ Modelos ML disponíveis para: {', '.join(supported_languages)}")
        except Exception as e:
            print(f"❌ Erro ao verificar modelos: {e}")
            dependencies_ok = False
        
        return dependencies_ok


def main():
    """Função principal CLI."""
    parser = argparse.ArgumentParser(
        description="Análise de vídeos para detecção de momentos virais",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main_analyzer.py video.mp4                    # Analisa um vídeo
  python main_analyzer.py --batch                      # Analisa todos os vídeos em downloads/
  python main_analyzer.py --batch --pattern "*.mov"    # Analisa arquivos .mov
  python main_analyzer.py video.mp4 --type podcast     # Analisa como podcast
  python main_analyzer.py video.mp4 --min-score 0.8   # Score mínimo 0.8
        """
    )
    
    parser.add_argument('video', nargs='?', help='Arquivo de vídeo para analisar')
    parser.add_argument('--batch', action='store_true', help='Processamento em lote')
    parser.add_argument('--input-dir', help='Diretório de entrada (padrão: downloads/)')
    parser.add_argument('--pattern', default='*.mp4', help='Padrão de arquivos (padrão: *.mp4)')
    parser.add_argument('--type', choices=['podcast', 'vlog', 'educational', 'debate', 'default'], 
                       default='default', help='Tipo de conteúdo')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Modelo Whisper (padrão: base)')
    parser.add_argument('--language', help='Força idioma específico (pt, en, etc.)')
    parser.add_argument('--min-score', type=float, default=0.6, 
                       help='Score mínimo para momentos virais (padrão: 0.6)')
    parser.add_argument('--validate', action='store_true', help='Apenas valida dependências')
    
    args = parser.parse_args()
    
    # Banner
    print("🎯 AUDIOANALYZER - DETECÇÃO DE MOMENTOS VIRAIS")
    print("=" * 60)
    
    try:
        # Inicializa pipeline
        pipeline = VideoAnalysisPipeline(
            whisper_model=args.whisper_model,
            content_type=args.type
        )
        
        # Validação de dependências
        if args.validate or not pipeline.validate_dependencies():
            if args.validate:
                print("\n✅ Validação concluída!")
            return
        
        # Processamento em lote
        if args.batch:
            result = pipeline.analyze_batch(
                input_directory=args.input_dir,
                file_pattern=args.pattern,
                min_viral_score=args.min_score
            )
            
            sys.exit(0 if result['success'] else 1)
        
        # Análise de vídeo único
        if not args.video:
            print("❌ Erro: Especifique um arquivo de vídeo ou use --batch")
            parser.print_help()
            sys.exit(1)
        
        result = pipeline.analyze_video(
            args.video,
            force_language=args.language,
            min_viral_score=args.min_score
        )
        
        sys.exit(0 if result['success'] else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Análise interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
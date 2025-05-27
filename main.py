#!/usr/bin/env python3
# 🚀 Main Pipeline - Projeto CORTE$
# Responsabilidade: Interface CLI e orquestração dos módulos

import sys
from pathlib import Path

# Importar módulos do projeto
from downloaders.video_downloader import YouTubeDownloader
from utils.file_manager import FileManager


def print_banner():
    """Exibe banner do projeto"""
    print("=" * 60)
    print("🎯 PROJETO CORTE$ - CORE-DOWNLOADER")
    print("   Máquina de Lucro com Cortes Virais")
    print("=" * 60)


def print_status(message, status="INFO"):
    """
    Imprime status formatado
    
    Args:
        message (str): Mensagem
        status (str): Tipo do status (INFO, SUCCESS, ERROR, WARNING)
    """
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "ERROR": "❌", 
        "WARNING": "⚠️"
    }
    
    icon = icons.get(status, "📋")
    print(f"{icon} {message}")


def validate_input(url):
    """
    Valida entrada do usuário
    
    Args:
        url (str): URL fornecida
        
    Returns:
        bool: True se válida
    """
    if not url or url.strip() == "":
        print_status("URL não pode estar vazia!", "ERROR")
        return False
        
    url = url.strip()
    downloader = YouTubeDownloader()
    
    if not downloader.validate_youtube_url(url):
        print_status("URL do YouTube inválida!", "ERROR")
        print_status("Exemplos válidos:", "INFO")
        print("  - https://www.youtube.com/watch?v=VIDEO_ID")
        print("  - https://youtu.be/VIDEO_ID")
        return False
        
    return True


def check_prerequisites():
    """
    Verifica pré-requisitos do sistema
    
    Returns:
        bool: True se tudo OK
    """
    print_status("Verificando pré-requisitos...", "INFO")
    
    # Verificar estrutura de diretórios
    if not FileManager.ensure_download_structure():
        print_status("Erro ao criar estrutura de diretórios", "ERROR")
        return False
    
    # Verificar espaço em disco
    space_info = FileManager.check_disk_space(".", 500)  # 500MB mínimo
    
    if not space_info.get('has_space', False):
        print_status(f"Espaço insuficiente! Necessário: 500MB, Disponível: {space_info.get('free_mb', 0):.1f}MB", "ERROR")
        return False
    
    print_status(f"Espaço disponível: {space_info['free_mb']:.1f}MB", "SUCCESS")
    return True


def download_video(url):
    """
    Executa download do vídeo
    
    Args:
        url (str): URL do YouTube
        
    Returns:
        bool: True se sucesso
    """
    print_status("Iniciando download...", "INFO")
    
    try:
        # Criar downloader
        downloader = YouTubeDownloader()
        
        # Obter informações do vídeo primeiro
        print_status("Obtendo informações do vídeo...", "INFO")
        video_info = downloader.get_video_info(url)
        
        if video_info:
            print_status(f"Título: {video_info['title']}", "INFO")
            print_status(f"Canal: {video_info.get('uploader', 'Desconhecido')}", "INFO")
            
            if video_info.get('duration'):
                duration_min = video_info['duration'] // 60
                duration_sec = video_info['duration'] % 60
                print_status(f"Duração: {duration_min}:{duration_sec:02d}", "INFO")
        
        # Executar download
        print_status("Baixando vídeo...", "INFO")
        result = downloader.download_video(url)
        
        if result['success']:
            print_status("Download concluído com sucesso!", "SUCCESS")
            print_status(f"Arquivo salvo: {result['file_path']}", "SUCCESS")
            
            # Mostrar informações do arquivo
            if result.get('file_size'):
                size_formatted = FileManager.format_file_size(result['file_size'])
                print_status(f"Tamanho: {size_formatted}", "INFO")
            
            return True
        else:
            print_status(f"Erro no download: {result['error']}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Erro inesperado: {str(e)}", "ERROR")
        return False


def main():
    """Função principal"""
    try:
        # Banner
        print_banner()
        
        # Verificar pré-requisitos
        if not check_prerequisites():
            sys.exit(1)
        
        print()
        
        # Loop principal
        while True:
            print_status("Digite a URL do vídeo do YouTube (ou 'quit' para sair):", "INFO")
            user_input = input("URL: ").strip()
            
            # Verificar se quer sair
            if user_input.lower() in ['quit', 'exit', 'q', 'sair']:
                print_status("Saindo... Até logo! 👋", "SUCCESS")
                break
            
            # Validar entrada
            if not validate_input(user_input):
                print()
                continue
            
            # Executar download
            success = download_video(user_input)
            
            print()
            
            if success:
                print_status("🎯 Vídeo pronto para processamento!", "SUCCESS")
                print_status("📋 Próximo módulo: AUDIO-ANALYZER", "INFO")
            else:
                print_status("Tente novamente com outra URL", "WARNING")
            
            print("-" * 60)
            print()
    
    except KeyboardInterrupt:
        print()
        print_status("Operação cancelada pelo usuário", "WARNING")
        sys.exit(0)
    except Exception as e:
        print_status(f"Erro crítico: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
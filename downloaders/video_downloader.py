# 📥 YouTube Downloader - Projeto CORTE$
# Responsabilidade: Download otimizado de vídeos do YouTube

import os
import re
import yt_dlp
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class YouTubeDownloader:
    """
    Classe para download otimizado de vídeos do YouTube
    
    Funcionalidades:
    - Validação de URLs do YouTube
    - Download em melhor qualidade disponível 
    - Padronização de nomes de arquivo
    - Organização em pasta downloads/
    """
    
    def __init__(self, download_dir="downloads"):
        """
        Inicializa o downloader
        
        Args:
            download_dir (str): Diretório onde salvar os vídeos
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Configurações do yt-dlp otimizadas para qualidade
        self.ydl_opts = {
            'format': 'best[height<=1080]',  # Melhor qualidade até 1080p
            'outtmpl': str(self.download_dir / '%(title)s.%(ext)s'),
            'noplaylist': True,  # Apenas o vídeo específico
            'extractaudio': False,  # Manter vídeo completo
            'writeinfojson': False,  # Não salvar metadados
            'writedescription': False,  # Não salvar descrição
            'writesubtitles': False,  # Não baixar legendas
        }
    
    def validate_youtube_url(self, url):
        """
        Valida se a URL é do YouTube
        
        Args:
            url (str): URL para validar
            
        Returns:
            bool: True se válida, False caso contrário
        """
        youtube_patterns = [
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
            r'(https?://)?(www\.)?youtu\.be/',
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        return False
    
    def extract_video_id(self, url):
        """
        Extrai o ID do vídeo da URL do YouTube
        
        Args:
            url (str): URL do YouTube
            
        Returns:
            str: ID do vídeo ou None se inválido
        """
        if not self.validate_youtube_url(url):
            return None
            
        # Padrões para extrair video ID
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def sanitize_filename(self, filename):
        """
        Remove caracteres inválidos do nome do arquivo
        
        Args:
            filename (str): Nome original
            
        Returns:
            str: Nome sanitizado
        """
        # Remove caracteres problemáticos
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Remove espaços extras
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        # Limita tamanho
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized
    
    def download_video(self, url, custom_filename=None):
        """
        Baixa vídeo do YouTube
        
        Args:
            url (str): URL do vídeo
            custom_filename (str): Nome personalizado (opcional)
            
        Returns:
            dict: Resultado do download com status e informações
        """
        try:
            # Validar URL
            if not self.validate_youtube_url(url):
                return {
                    'success': False,
                    'error': 'URL do YouTube inválida',
                    'file_path': None
                }
            
            # Extrair informações básicas primeiro
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                try:
                    info = ydl.extract_info(url, download=False)
                    video_title = info.get('title', 'video_sem_titulo')
                    video_id = info.get('id', 'unknown')
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Erro ao extrair informações: {str(e)}',
                        'file_path': None
                    }
            
            # Preparar nome do arquivo
            if custom_filename:
                filename = self.sanitize_filename(custom_filename)
            else:
                filename = self.sanitize_filename(video_title)
            
            # Atualizar configurações com nome personalizado
            opts = self.ydl_opts.copy()
            opts['outtmpl'] = str(self.download_dir / f'{filename}.%(ext)s')
            
            # Fazer download
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            
            # Encontrar arquivo baixado
            possible_extensions = ['mp4', 'mkv', 'webm', 'avi']
            downloaded_file = None
            
            for ext in possible_extensions:
                file_path = self.download_dir / f'{filename}.{ext}'
                if file_path.exists():
                    downloaded_file = file_path
                    break
            
            if downloaded_file:
                return {
                    'success': True,
                    'error': None,
                    'file_path': str(downloaded_file),
                    'video_title': video_title,
                    'video_id': video_id,
                    'file_size': downloaded_file.stat().st_size
                }
            else:
                return {
                    'success': False,
                    'error': 'Arquivo baixado não encontrado',
                    'file_path': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Erro no download: {str(e)}',
                'file_path': None
            }
    
    def get_video_info(self, url):
        """
        Obtém informações do vídeo sem baixar
        
        Args:
            url (str): URL do vídeo
            
        Returns:
            dict: Informações do vídeo
        """
        try:
            if not self.validate_youtube_url(url):
                return None
                
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'video_id': info.get('id'),
                    'thumbnail': info.get('thumbnail')
                }
        except Exception as e:
            print(f"Erro ao obter informações: {e}")
            return None


# Função de conveniência para uso direto
def download_youtube_video(url, download_dir="downloads", custom_filename=None):
    """
    Função simples para download direto
    
    Args:
        url (str): URL do YouTube
        download_dir (str): Pasta de destino
        custom_filename (str): Nome personalizado
        
    Returns:
        dict: Resultado do download
    """
    downloader = YouTubeDownloader(download_dir)
    return downloader.download_video(url, custom_filename)
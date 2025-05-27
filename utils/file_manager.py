# 🗂️ File Manager - Projeto CORTE$
# Responsabilidade: Operações essenciais de arquivo e organização

import os
import shutil
import re
from pathlib import Path
from typing import Optional, List


class FileManager:
    """
    Classe para operações essenciais de arquivo
    
    Funcionalidades:
    - Criação segura de diretórios
    - Validação de espaço em disco
    - Limpeza de nomes de arquivo
    - Verificação de arquivos existentes
    """
    
    @staticmethod
    def create_directory(path, exist_ok=True):
        """
        Cria diretório de forma segura
        
        Args:
            path (str): Caminho do diretório
            exist_ok (bool): Se True, não gera erro se já existir
            
        Returns:
            bool: True se criado/existir, False se erro
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=exist_ok)
            return True
        except Exception as e:
            print(f"Erro ao criar diretório {path}: {e}")
            return False
    
    @staticmethod
    def clean_filename(filename):
        """
        Limpa nome de arquivo removendo caracteres inválidos
        
        Args:
            filename (str): Nome original
            
        Returns:
            str: Nome limpo e seguro
        """
        # Remove caracteres inválidos para sistemas de arquivo
        cleaned = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Remove múltiplos espaços
        cleaned = re.sub(r'\s+', '_', cleaned)
        
        # Remove pontos no início/fim
        cleaned = cleaned.strip('.')
        
        # Limita tamanho do nome
        if len(cleaned) > 100:
            name, ext = os.path.splitext(cleaned)
            cleaned = name[:100-len(ext)] + ext
        
        return cleaned if cleaned else "arquivo_sem_nome"
    
    @staticmethod
    def check_disk_space(path, required_mb=100):
        """
        Verifica espaço disponível em disco
        
        Args:
            path (str): Caminho para verificar
            required_mb (int): Espaço mínimo necessário em MB
            
        Returns:
            dict: Informações sobre espaço em disco
        """
        try:
            stats = shutil.disk_usage(path)
            free_mb = stats.free / (1024 * 1024)
            total_mb = stats.total / (1024 * 1024)
            
            return {
                'has_space': free_mb >= required_mb,
                'free_mb': round(free_mb, 2),
                'total_mb': round(total_mb, 2),
                'required_mb': required_mb
            }
        except Exception as e:
            print(f"Erro ao verificar espaço: {e}")
            return {
                'has_space': False,
                'error': str(e)
            }
    
    @staticmethod
    def file_exists(file_path):
        """
        Verifica se arquivo existe
        
        Args:
            file_path (str): Caminho do arquivo
            
        Returns:
            bool: True se existir
        """
        return Path(file_path).exists()
    
    @staticmethod
    def get_file_size(file_path):
        """
        Obtém tamanho do arquivo em bytes
        
        Args:
            file_path (str): Caminho do arquivo
            
        Returns:
            int: Tamanho em bytes, ou None se erro
        """
        try:
            return Path(file_path).stat().st_size
        except Exception:
            return None
    
    @staticmethod
    def format_file_size(size_bytes):
        """
        Formata tamanho de arquivo em formato legível
        
        Args:
            size_bytes (int): Tamanho em bytes
            
        Returns:
            str: Tamanho formatado (ex: "1.5 MB")
        """
        if size_bytes is None:
            return "Desconhecido"
            
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def list_files_by_extension(directory, extension):
        """
        Lista arquivos por extensão em um diretório
        
        Args:
            directory (str): Caminho do diretório
            extension (str): Extensão desejada (ex: ".mp4")
            
        Returns:
            List[str]: Lista de caminhos dos arquivos
        """
        try:
            path = Path(directory)
            if not path.exists():
                return []
                
            pattern = f"*{extension}"
            return [str(file) for file in path.glob(pattern)]
        except Exception as e:
            print(f"Erro ao listar arquivos: {e}")
            return []
    
    @staticmethod
    def ensure_download_structure():
        """
        Garante que a estrutura básica de pastas existe
        
        Returns:
            bool: True se estrutura foi criada/existe
        """
        directories = [
            "downloads",
            "downloads/temp",
            "logs"
        ]
        
        success = True
        for directory in directories:
            if not FileManager.create_directory(directory):
                success = False
                
        return success
    
    @staticmethod
    def get_unique_filename(file_path):
        """
        Gera nome único se arquivo já existir
        
        Args:
            file_path (str): Caminho do arquivo desejado
            
        Returns:
            str: Caminho único (adiciona número se necessário)
        """
        path = Path(file_path)
        
        if not path.exists():
            return str(path)
        
        # Separar nome e extensão
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        
        # Procurar nome disponível
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            
            if not new_path.exists():
                return str(new_path)
                
            counter += 1
            
            # Limite de segurança
            if counter > 1000:
                break
        
        return str(path)


# Funções de conveniência para uso direto
def ensure_directories():
    """Cria estrutura básica de diretórios"""
    return FileManager.ensure_download_structure()

def clean_name(filename):
    """Limpa nome de arquivo"""
    return FileManager.clean_filename(filename)

def check_space(required_mb=100):
    """Verifica espaço em disco"""
    return FileManager.check_disk_space(".", required_mb)
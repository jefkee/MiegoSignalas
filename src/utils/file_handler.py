import os
import shutil

class FileHandler:
    """Handles file operations for the application"""
    
    @staticmethod
    def ensure_dir(directory):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    @classmethod
    def save_file(cls, file, upload_dir):
        """Save uploaded file to directory
        
        Args:
            file: File object to save
            upload_dir: Directory to save the file in
            
        Returns:
            str: Path to saved file
        """
        try:
            # Ensure directory exists
            cls.ensure_dir(upload_dir)
            
            # Save file
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            
            return file_path
            
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise
            
    @staticmethod
    def cleanup_files(directory, pattern=None):
        """Clean up files in directory
        
        Args:
            directory: Directory to clean up
            pattern: Optional pattern to match files to delete
        """
        try:
            if os.path.exists(directory):
                if pattern:
                    # Remove only files matching pattern
                    for file in os.listdir(directory):
                        if pattern in file:
                            os.remove(os.path.join(directory, file))
                else:
                    # Remove entire directory
                    shutil.rmtree(directory)
                    
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")
            raise
            
    @staticmethod
    def get_file_list(directory, pattern=None):
        """Get list of files in directory
        
        Args:
            directory: Directory to list files from
            pattern: Optional pattern to match files
            
        Returns:
            list: List of file paths
        """
        try:
            if not os.path.exists(directory):
                return []
                
            if pattern:
                return [f for f in os.listdir(directory) if pattern in f]
            else:
                return os.listdir(directory)
                
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            raise 
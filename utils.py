import os

def list_files_and_write_contents(dir_path, output_file, max_files=10):
    # List all files and directories in the given path
    files = os.listdir(dir_path)
    # Filter out only files (not directories)
    files = [f for f in files if os.path.isfile(os.path.join(dir_path, f))][:max_files]
    
    # Open the output file in write mode
    with open(output_file, 'w') as out_file:
        for file in files:
            file_path = os.path.join(dir_path, file)
            
            # Read the contents of the file and write them into the output file
            with open(file_path, 'r') as in_file:
                content = in_file.read()
                out_file.write(content)
            

# Example usage
list_files_and_write_contents('/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/embd/gold_std/', 'data/output_10.txt', max_files=10)

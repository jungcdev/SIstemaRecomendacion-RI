# Definir las rutas de los archivos de entrada y salida
input_file_path = 'datasets/ml-1m/movies.dat'
output_file_path = 'datasets/ml-1m/movies.csv'

# Leer el contenido del archivo de entrada, reemplazar los delimitadores y guardar en el archivo de salida
with open(input_file_path, 'r', encoding='utf-8', errors='replace') as infile:
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            new_line = line.replace('::', ',')
            outfile.write(new_line)

print(f"Archivo convertido y guardado en {output_file_path}")
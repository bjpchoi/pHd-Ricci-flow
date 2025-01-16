import csv
from collections import Counter


def get_csv_dimensions(file_path):
   with open(file_path, mode='r', encoding='utf-8-sig') as file:
       reader = csv.reader(file)
       rows = list(reader)
       num_rows = len(rows)
       num_columns = len(rows[0]) if rows else 0
   return num_rows, num_columns


def count_first_column_entries(file_path):
   counts = Counter()
   with open(file_path, mode='r', encoding='utf-8-sig') as file:
       reader = csv.reader(file)
       for row in reader:
           if row:
               try:
                   value = int(row[0])
                   if -1 <= value <= 9:
                       counts[value] += 1
               except ValueError:
                   continue
   return counts


def compute_digit_distribution(file_path):
   with open(file_path, mode='r', encoding='utf-8-sig') as file:
       reader = csv.reader(file)
       next(reader)  
       first_column_values = [row[0] for row in reader if row]
   return Counter(first_column_values)




def process_csv(input_file_path, output_file_path, negative_one_count, other_count):
   with open(input_file_path, mode='r', encoding='utf-8-sig') as infile:
       reader = csv.reader(infile)
       rows = list(reader)


       header = rows[0]
       data = rows[1:]


       negative_one_rows = [row for row in data if row[0] == '-1']
       other_rows = [row for row in data if row[0] != '-1']


       if len(negative_one_rows) < negative_one_count:
           raise ValueError(f"Not enough rows starting with -1 in {input_file_path}")
       if len(other_rows) < other_count:
           raise ValueError(f"Not enough rows not starting with -1 in {input_file_path}")


       selected_negative_one_rows = random.sample(negative_one_rows, negative_one_count)
       selected_other_rows = random.sample(other_rows, other_count)


       selected_rows = [header] + selected_negative_one_rows + selected_other_rows


   with open(output_file_path, mode='w', newline='', encoding='utf-8-sig') as outfile:
       writer = csv.writer(outfile)
       writer.writerows(selected_rows)
   print(f"Processed {input_file_path} and saved to {output_file_path}")


def rename_and_download_files():
   train_source_path = 'drive/MyDrive/number_train.csv'
   test_source_path = 'drive/MyDrive/number_test.csv'
   train_dest_path = '/content/num_train.csv'
   test_dest_path = '/content/num_test.csv'


   shutil.move(train_source_path, train_dest_path)
   shutil.move(test_source_path, test_dest_path)


   from google.colab import files
   files.download(train_dest_path)
   files.download(test_dest_path)
   print("Files renamed and downloaded successfully.")


def process_and_download_files():
   # File paths
   train_source_path = '/content/num_train.csv'
   test_source_path = '/content/num_test.csv'
   train_dest_path = '/content/numerical_train.csv'
   test_dest_path = '/content/numerical_test.csv'


   # Helper function to modify labels
   def modify_labels_and_save(input_path, output_path):
       with open(input_path, mode='r', encoding='utf-8-sig') as infile:
           reader = csv.reader(infile)
           rows = list(reader)


       header = rows[0]
       data = rows[1:]
       modified_data = [[('1' if row[0] != '-1' else '-1')] + row[1:] for row in data]


       with open(output_path, mode='w', newline='', encoding='utf-8-sig') as outfile:
           writer = csv.writer(outfile)
           writer.writerow(header)
           writer.writerows(modified_data)


   modify_labels_and_save(train_source_path, train_dest_path)
   modify_labels_and_save(test_source_path, test_dest_path)


   from google.colab import files
   files.download(train_dest_path)
   files.download(test_dest_path)
   print("Files processed, renamed, and downloaded successfully.")


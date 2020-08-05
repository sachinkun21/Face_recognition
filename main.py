from src.load_data import save_array,load_array
from src.embeddings import create_embeddings


train_path = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/train/'
test_path = '/Users/sachinkun21/PycharmProjects/FaceRecog/dataset/val/'
input_path = 'dataset/saved_arrays/faces-dataset.npz'

#save_array(train_path,test_path,path_to_save = array_path)

output_path = 'dataset/saved_arrays/embeddings-dataset.npz'

create_embeddings(input_path, output_path)








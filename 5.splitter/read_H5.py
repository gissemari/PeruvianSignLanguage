import h5py
import numpy as np

def read_lsa64_hdf5_dataset(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        # Accede a los grupos de entrenamiento y validación
        train_group = h5_file['LSA64']
        #validation_group = h5_file['LSA64']['validation']

        # Accede a los conjuntos de datos en cada grupo
        train_data = train_group['data']
        train_label = train_group['label']
        train_length = train_group['length']
        train_class_number = train_group['class_number']
        train_shape = train_group['shape'][:]
        #validation_data = validation_group['data']
        #validation_label = validation_group['label']
        #validation_class_number = validation_group['class_number']

        # Itera sobre los elementos en los conjuntos de entrenamiento
        for i in range(len(train_data)):
            data_i_flat = np.array(train_data[i]) 
            data_shape = (train_length[i], train_shape[0], train_shape[1])  # Restaurar la forma original
            data_i = data_i_flat.reshape(data_shape)
            
            label_i = train_label[i]
            class_number_i = train_class_number[i]

            # Hacer algo con los datos leídos (imprimir por ejemplo)
            print(f'Datos de entrenamiento {i}: {data_i.shape}|{train_shape}, Etiqueta: {label_i}, Número de clase: {class_number_i}')

        # Itera sobre los elementos en los conjuntos de validación
        '''
        for i in range(len(validation_data)):
            data_i_flat = np.array(validation_data[i])  # Asegúrate de convertir a un array NumPy
            data_shape = (data_i_flat.shape[0] // 2, 2, 544)  # Restaurar la forma original
            data_i = data_i_flat.reshape(data_shape)
            
            label_i = validation_label[i]
            class_number_i = validation_class_number[i]

            # Hacer algo con los datos leídos (imprimir por ejemplo)
            print(f'Datos de validación {i}: {data_i.shape}, Etiqueta: {label_i}, Número de clase: {class_number_i}')
        '''
# Uso del ejemplo
read_lsa64_hdf5_dataset('../Data/LSA64/LSA64--64--mediapipe-train.hdf5')

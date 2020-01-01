. vtck.config
python make_dataset_vtck.py $data_root_dir $h5py_path $train_proportion
python make_single_samples.py $h5py_path $index_path $n_samples $seg_len $speaker_used_path

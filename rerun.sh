# python tdmil/train.py --config_file config/property_img_mil_numpygenerator.json
# python tdmil/eval.py --checkpoint_path model_weights/property_img_mil_numpygenerator/ --inp_csv  datasets/train_image_emb.csv --num_rerun 1
# python tdmil/eval.py --checkpoint_path model_weights/property_img_mil_numpygenerator/ --inp_csv  datasets/val_image_emb.csv --num_rerun 1
# python tdmil/eval.py --checkpoint_path model_weights/property_img_transformer_numpygenerator_t2/ --inp_csv  datasets/train_image_emb.csv --num_rerun 1
# python tdmil/eval.py --checkpoint_path model_weights/property_img_transformer_numpygenerator_t2/ --inp_csv  datasets/val_image_emb.csv --num_rerun 1
# python tdmil/train.py --config_file config/property_img_transformer_numpygenerator_t2.json
python tdmil/train.py --config_file config/mnst_transformer_target_0.json
python tdmil/train.py --config_file config/mnst_transformer_target_1.json
python tdmil/train.py --config_file config/mnst_transformer_target_2.json
